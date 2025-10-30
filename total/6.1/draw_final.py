import numpy as np
try:
    from scipy.interpolate import PchipInterpolator  # 单调保持的样条插值
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
try:
    from scipy.signal import savgol_filter
    _HAS_SG = True
except Exception:
    _HAS_SG = False
import re
import sys
import argparse
import datetime
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


def parse_iso_ts(ts_str: str) -> Optional[datetime.datetime]:
    """
    将 ISO 时间字符串解析为 datetime 对象，处理 Z 时区表示。
    """
    if not ts_str:
        return None
    ts_str = ts_str.strip()
    # 将 Z 转换为 +00:00，便于 fromisoformat 解析
    ts_str = ts_str.replace('Z', '+00:00')
    try:
        return datetime.datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def extract_timestamp_and_cov(line: str, output_type: str) -> Optional[Tuple[datetime.datetime, int]]:
    """
    从一行日志中尝试提取时间戳和覆盖率。
    - 对于前两份日志：使用 cal/max cover 的前一项
    - 对于第三份日志：使用 cover 的数值
    返回 (timestamp, coverage) 或 None
    """
    line = line.strip()
    if not line:
        return None

    ts: Optional[datetime.datetime] = None
    cov: Optional[int] = None

    # 尝试解析时间戳：
    # 1) [TIMESTAMP INFO ]
    m = re.match(r'\[(.*?)\sINFO', line)
    if m:
        ts_str = m.group(1)
        ts = parse_iso_ts(ts_str)

    # 2) 常见的 "YYYY/MM/DD HH:MM:SS" 格式
    if ts is None:
        m = re.match(r'(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', line)
        if m:
            ts_str = m.group(1)
            try:
                ts = datetime.datetime.strptime(ts_str, '%Y/%m/%d %H:%M:%S')
            except ValueError:
                ts = None

    if ts is None:
        return None  # 无法解析时间

    # 尝试解析覆盖率：
    if output_type == 'cov':
        m = re.search(r'cal/max cover\s+(\d+)\s*/\s*(\d+)', line)
        if m:
            cov = int(m.group(1))
            return ts, cov
        else:
            m = re.search(r'coverage=(\d+)', line)
            if m:
                cov = int(m.group(1))
                return ts, cov
            else:
                m = re.search(r'cover (\d+)', line)
                if m:
                    cov = int(m.group(1))
                    return ts, cov
    elif output_type == 'corpus':
        m = re.search(r'corpus:\s+(\d+)', line)
        if m:
            cov = int(m.group(1))
            return ts, cov
    elif output_type == 'exec':
        m = re.search(r'exec:\s+(\d+)', line)
        if m:
            cov = int(m.group(1))
            return ts, cov

    return None


def read_log(path: str, output_type: str) -> List[Tuple[datetime.datetime, int]]:
    """
    读取日志文件，返回按时间排序的 (timestamp, coverage) 列表
    只要能够解析出 timestamp 与 coverage 就会被保留。
    """
    entries: List[Tuple[datetime.datetime, int]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            res = extract_timestamp_and_cov(line, output_type)
            if res is not None:
                entries.append(res)
    # 按时间排序，防止文件内乱序
    entries.sort(key=lambda x: x[0])
    return entries


def to_elapsed_hours(entries: List[Tuple[datetime.datetime, int]]) -> Tuple[List[float], List[int], datetime.datetime]:
    """
    将原始时间转换为相对起点的经过小时数，以及对应的覆盖率值。
    返回 (x_hours, y_cov, t0)
    """
    if not entries:
        return [], [], None

    t0 = entries[0][0]
    x_hours: List[float] = []
    y_cov: List[int] = []
    for ts, cov in entries:
        delta = ts - t0
        hours = delta.total_seconds() / 3600.0
        x_hours.append(hours)
        y_cov.append(cov)
    return x_hours, y_cov, t0

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """
    移动平均，长度保持与输入一致；并用最大累积确保不下降。
    兼容奇偶窗口。
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if window <= 1 or n == 0:
        return y

    # 计算左右填充，使 valid 卷积的输出长度恰好为 n
    left = window // 2
    right = window - 1 - left
    y_pad = np.pad(y, (left, right), mode='edge')

    kernel = np.ones(window, dtype=float) / window
    y_ma = np.convolve(y_pad, kernel, mode='valid')  # 长度 = n

    # 保证非降序（覆盖率不随时间下降）
    y_ma = np.maximum.accumulate(y_ma)
    return y_ma

def smooth_xy(x, y,
              resample_step_min: float = 1.0,
              ma_minutes: float = 10.0):
    """
    不均匀采样数据的重采样 + 平滑。
    返回 (x_new, y_smooth)，两者长度严格一致。
    """
    if not x or not y:
        return np.array([]), np.array([])

    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    # 保证非降序
    y_arr = np.maximum.accumulate(y_arr)

    step_h = resample_step_min / 60.0
    x_min, x_max = 0.0, float(x_arr[-1])
    # 包含末端，使 0..24 小时步长为 1 分钟时得到 1441 个点
    x_new = np.arange(x_min, x_max + 1e-12, step_h)

    if _HAS_SCIPY:
        interp = PchipInterpolator(x_arr, y_arr, extrapolate=False)
        y_new = interp(x_new)
    else:
        y_new = np.interp(x_new, x_arr, y_arr)

    # 将 NaN（若存在）用前向填充/端点填充处理
    y_new = np.asarray(y_new, dtype=float)
    if np.isnan(y_new).any():
        # 前向填充
        idx = np.where(np.isnan(y_new))[0]
        for i in idx:
            y_new[i] = y_new[i-1] if i > 0 else y_arr[0]

    # 计算移动平均窗口（分钟）
    window = max(1, int(round(ma_minutes / resample_step_min)))
    y_smooth = moving_average(y_new, window)

    # 保险：长度一致化
    m = min(x_new.shape[0], y_smooth.shape[0])
    x_new = x_new[:m]
    y_smooth = y_smooth[:m]
    return x_new, y_smooth

def strong_smooth_monotone(y: np.ndarray,
                           step_min: float,
                           win_min: float = 90.0,
                           passes: int = 2) -> np.ndarray:
    """
    仅用于红线：更强的平滑。
    - step_min: 重采样步长（分钟），用于把窗口换算成样本长度
    - win_min: 主要平滑窗口（分钟），建议 60~120
    - passes: 平滑次数，建议 2~3
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    w = max(3, int(round(win_min / step_min)))
    if w % 2 == 0:
        w += 1  # 一些算法要求奇数窗口

    ys = y.copy()
    if _HAS_SG and w >= 5:
        # Savitzky–Golay + 单调性修正，重复多次得到更“圆滑”的曲线
        for _ in range(passes):
            ys = savgol_filter(ys, window_length=w, polyorder=3, mode='interp')
            ys = np.maximum.accumulate(ys)
    else:
        # 没有 SciPy：用更长窗口的移动平均，多次迭代
        for _ in range(passes * 2):
            ys = moving_average(ys, w)
            ys = np.maximum.accumulate(ys)
    return ys

def plot_coverage(log_paths: List[str], labels: List[str], output_type: str, out_path: Optional[str] = None):
    # 读取日志
    data = []
    for i, path in enumerate(log_paths):
        entries = read_log(path, output_type)
        x, y, _t0 = to_elapsed_hours(entries)
        data.append((x, y))

    # 数据只画到 24 小时；坐标轴显示到 25 小时
    H_CUTOFF = 24.0      # 数据截断
    H_AXIS_MAX = 25.0    # 坐标轴最大值（用于显示刻度到 25）

    # 平滑参数
    RESAMPLE_STEP_MIN = 1.0   # 重采样步长（分钟）
    MA_MINUTES = 10.0         # 移动平均窗口（分钟）

    data_smooth = []
    for i, (x, y) in enumerate(data):
        if x:
            last_idx = np.searchsorted(np.array(x), H_CUTOFF, side='right')
            x_cut = x[:last_idx]
            y_cut = y[:last_idx]
        else:
            x_cut, y_cut = [], []

        # 先按通用参数平滑（不改动其它曲线的默认外观）
        x_sm, y_sm = smooth_xy(
            x_cut, y_cut,
            resample_step_min=RESAMPLE_STEP_MIN,
            ma_minutes=MA_MINUTES
        )

        # 仅对红线（KConfigFuzz 或包含 'kconfig' 的标签）做更强的平滑
        lab = labels[i] if i < len(labels) else ""
        if lab == 'KConfigFuzz' or ('kconfig' in lab.lower()):
            # “最大可能”地平滑：较长窗口 + 多次
            y_sm = strong_smooth_monotone(
                y_sm,
                step_min=RESAMPLE_STEP_MIN,
                win_min=90.0,     # 你可以尝试 60~120，越大越“圆滑”
                passes=2          # 2~3 次
            )

        # 保证长度一致并只保留到 24 小时
        n = min(len(x_sm), len(y_sm))
        x_sm, y_sm = x_sm[:n], y_sm[:n]
        mask = (x_sm <= H_CUTOFF)
        data_smooth.append((x_sm[mask], y_sm[mask]))

    # y 轴最大值（用于设定 ylim）
    all_y_flat = [v for _, yv in data_smooth for v in yv] if data_smooth else []
    y_max = max(all_y_flat) if all_y_flat else 1

    # ---------- 目标时刻与差异计算 ----------
    # 计算每条曲线最后的时间（小时）
    last_times = [(x[-1] if x else 0.0) for x, _ in data]

    TARGET_FULL = 24.0
    min_last_time = min(last_times) if last_times else 0.0
    if min_last_time >= TARGET_FULL:
        target = TARGET_FULL
        header = "满24小时的覆盖率差异"
    else:
        target = min_last_time
        header = f"在第{target:.2f}小时的覆盖率差异"

    # 辅助：在 time t 取序列值（取最后一个 x <= t 的 y）
    def value_at(x_list: List[float], y_list: List[int], t: float) -> Optional[float]:
        if not x_list:
            return None
        idx = None
        for i, xv in enumerate(x_list):
            if xv <= t + 1e-9:
                idx = i
            else:
                break
        if idx is None:
            # 所有点都大于 t，返回第一个点
            return float(y_list[0]) if y_list else None
        return float(y_list[idx])

    # 找到 KConfigFuzz 的索引（优先精确匹配）
    kidx = None
    for i, lab in enumerate(labels):
        if lab == 'KConfigFuzz':
            kidx = i
            break
    if kidx is None:
        for i, lab in enumerate(labels):
            if 'KConfig' in lab or 'kconfig' in lab.lower():
                kidx = i
                break

    if kidx is None:
        print("警告：未找到名为 'KConfigFuzz' 或包含 'KConfig' 的标签，跳过差异计算。")
    else:
        xk, yk = data[kidx]
        k_val = value_at(xk, yk, target)
        if k_val is None:
            print("警告：KConfigFuzz 在目标时刻没有数据，跳过差异计算。")
        else:
            print(f"\n{header}（目标时刻 = {target:.2f} 小时）：")
            for i, lab in enumerate(labels):
                if i == kidx:
                    continue
                xi, yi = data[i]
                oth_val = value_at(xi, yi, target)
                if oth_val is None:
                    print(f"  {lab}: 在目标时刻无数据，跳过。")
                    continue
                if abs(oth_val) < 1e-9:
                    diff_abs = k_val - oth_val
                    print(f"  {lab}: 其它为0，绝对差 = {diff_abs:.2f}（KConfigFuzz={k_val:.2f}）")
                else:
                    pct = (k_val - oth_val) / oth_val * 100.0
                    trend = "高" if pct >= 0 else "低"
                    print(f"  相对于 {lab}: KConfigFuzz 比它 {trend} {abs(pct):.2f}% （K={k_val:.2f}, {lab}={oth_val:.2f}）")

    # 画图
    plt.figure(figsize=(8, 5))
    plt.grid(True, linestyle='-', linewidth=0.6, alpha=0.15)

    # 更细的线宽，并区分 HFL 与 KConfigFuzz
    style_map = {
        'KConfigFuzz':          dict(color='red',    linestyle='-',  linewidth=1.6),
        'Syzkaller':       dict(color='green',  linestyle='--', linewidth=1.6),
        'HFL':             dict(color='orange', linestyle='-.', linewidth=1.6),
        'HEALER':     dict(color='blue',   linestyle='-.',  linewidth=1.6),
        'KConfigFuzz-D':    dict(color='purple',   linestyle=':',  linewidth=1.6),
        'KConfigFuzz-V':    dict(color='deepskyblue',   linestyle=':',  linewidth=1.6),
    }

    for (x_sm, y_sm), lab in zip(data_smooth, labels):
        style = style_map.get(lab, dict(color=None, linestyle='-', linewidth=1.6))
        plt.plot(
            x_sm, y_sm,
            label=lab,
            **style,
            solid_capstyle='round',
            solid_joinstyle='round'
        )

    # 坐标轴与标题
    plt.xlabel('time(hour)')
    plt.ylabel('codeblock coverage' if output_type == 'cov' else (output_type + ' num'))
    plt.title('Linux v6.1')

    # 坐标轴范围与刻度：数据到 24，但刻度显示到 25
    plt.xlim(0, H_AXIS_MAX)
    plt.xticks([0, 5, 10, 15, 20, 25])
    plt.ylim(0, max(1.0, y_max * 1.05))

    plt.legend(loc='lower right', frameon=True, fancybox=True, framealpha=1.0, borderpad=0.6)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"覆盖率图已保存至 {out_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="绘制三份日志的覆盖率随时间变化图（起点对齐）")
    parser.add_argument('--files', nargs='+', help="日志文件")
    parser.add_argument('--labels', nargs='+', help="标签")
    parser.add_argument('--type', help="类型")
    parser.add_argument('--out', default=None, help='输出图片路径（可选）')

    args = parser.parse_args()

    if args.type not in ["cov", "corpus", "exec"]:
        print("Invalid output type.")

    plot_coverage(
        log_paths=args.files,
        labels=args.labels,
        output_type=args.type,
        out_path=args.out
    )


if __name__ == '__main__':
    main()

'''
调用命令：
python3 draw2.py --file1=log_healer_6.2gcc_with_config_static --file2=log_healer_6.2gcc --file3=无syscallpair的24小时.txt --label1=HEALER_Static --label2=HEALER --label3=Syzkaller --out=fuzz_coverage_comparison.png
'''