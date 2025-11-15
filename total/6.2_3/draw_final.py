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

def detect_file_type(path: str) -> str:
    """
    检测文件类型：'log' 表示日志文件，'avg' 表示平均数据文件
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 读取前几行来判断文件类型
            lines = []
            for i, line in enumerate(f):
                if i >= 5:  # 只读取前5行
                    break
                lines.append(line.strip())
            
            # 检查是否为平均数据文件格式
            if lines and lines[0].startswith('elapsed_hours,'):
                return 'avg'
            
            # 检查是否包含日志格式的特征
            for line in lines:
                if line.startswith('[') and 'INFO' in line:
                    return 'log'
                if re.match(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}', line):
                    return 'log'
            
            # 默认认为是日志文件
            return 'log'
    except Exception:
        return 'log'

def read_file_auto(path: str, output_type: str) -> List[Tuple]:
    """
    自动检测文件类型并读取数据
    返回 [(x_value, y_value), ...] 列表
    """
    file_type = detect_file_type(path)
    
    if file_type == 'avg':
        # 读取平均数据文件
        entries: List[Tuple[float, int]] = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) <= 1:  # 只有标题或空文件
                    return entries
                
                # 跳过标题行
                for line in lines[1:]:
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) == 2:
                            try:
                                elapsed_hours = float(parts[0])
                                value = float(parts[1])
                                entries.append((elapsed_hours, int(value)))
                            except ValueError:
                                continue
        except Exception as e:
            print(f"读取文件 {path} 时出错: {e}")
            return []
        return entries
    else:
        # 读取日志文件
        log_entries = read_log(path, output_type)
        x_hours, y_values, _ = to_elapsed_hours(log_entries)
        return list(zip(x_hours, y_values))

def read_log(path: str, output_type: str) -> List[Tuple[datetime.datetime, int]]:
    """
    读取日志文件，返回按时间排序的 (timestamp, coverage) 列表
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

def align_and_average_data(log_paths: List[str], output_type: str) -> List[Tuple[float, float]]:
    """
    对多个日志文件的数据进行时间对齐并计算平均值
    只对所有文件都有的时间段做平均
    返回 [(elapsed_hours, average_value), ...]
    """
    # 读取所有日志文件的数据
    all_data = []
    for path in log_paths:
        entries = read_log(path, output_type)
        x_hours, y_values, t0 = to_elapsed_hours(entries)
        if x_hours and y_values:
            all_data.append(list(zip(x_hours, y_values)))
    
    if not all_data:
        return []
    
    # 找到所有文件都有的时间段范围
    # 获取每个文件的时间范围
    time_ranges = [(data[0][0], data[-1][0]) for data in all_data if data]
    
    if not time_ranges:
        return []
    
    # 找到交集时间段
    common_start = max(range_[0] for range_ in time_ranges)
    common_end = min(range_[1] for range_ in time_ranges)
    
    if common_start >= common_end:
        return []
    
    # 只保留交集时间段内的数据
    trimmed_data = []
    for data in all_data:
        trimmed = [(x, y) for x, y in data if common_start <= x <= common_end]
        if trimmed:
            trimmed_data.append(trimmed)
    
    if not trimmed_data:
        return []
    
    # 按时间点进行插值平均
    # 使用第一个文件的时间点作为参考点
    reference_times = [point[0] for point in trimmed_data[0]]
    
    averaged_data = []
    for time_point in reference_times:
        values = []
        for data_series in trimmed_data:
            # 找到最接近的时间点的值（线性插值）
            value = interpolate_value(data_series, time_point)
            if value is not None:
                values.append(value)
        
        if len(values) == len(trimmed_data) and values:  # 所有文件都有对应值
            avg_value = sum(values) / len(values)
            averaged_data.append((time_point, avg_value))
    
    return averaged_data

def interpolate_value(data_series: List[Tuple[float, int]], target_time: float) -> Optional[float]:
    """
    在数据序列中找到目标时间点的插值
    """
    if not data_series:
        return None
    
    # 如果目标时间小于第一个点或大于最后一个点，返回None
    if target_time < data_series[0][0] or target_time > data_series[-1][0]:
        return None
    
    # 找到相邻的两个点进行线性插值
    for i in range(len(data_series) - 1):
        t1, v1 = data_series[i]
        t2, v2 = data_series[i + 1]
        
        if t1 <= target_time <= t2:
            if t1 == t2:
                return float(v1)
            # 线性插值
            ratio = (target_time - t1) / (t2 - t1)
            interpolated_value = v1 + ratio * (v2 - v1)
            return interpolated_value
    
    return None

def save_averaged_data(averaged_data: List[Tuple[float, float]], output_path: str):
    """
    将平均后的数据保存到文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("elapsed_hours,average_value\n")
        for hours, value in averaged_data:
            f.write(f"{hours:.6f},{value:.2f}\n")
    print(f"平均数据已保存至 {output_path}")

def plot_coverage(all_files: List[str], labels: List[str], output_type: str, out_path: Optional[str] = None):
    """
    绘制曲线并输出图像。
    自动检测并处理不同类型的文件
    """
    # 读取所有文件数据
    data = []
    display_labels = []
    
    for i, path in enumerate(all_files):
        # 自动检测并读取文件
        entries = read_file_auto(path, output_type)
        if entries:
            x_vals = [entry[0] for entry in entries]
            y_vals = [entry[1] for entry in entries]
            data.append((x_vals, y_vals))
            if i < len(labels):
                display_labels.append(labels[i])
            else:
                display_labels.append(f"File {i+1}")
    
    if not data:
        print("没有数据可绘制")
        return

    # 数据只画到 24 小时；坐标轴显示到 25 小时
    H_CUTOFF = 24.0      # 数据截断
    H_AXIS_MAX = 25.0    # 坐标轴最大值（用于显示刻度到 25）
    
    # 计算图形上的最大 y 值
    all_y = [yv for _, yv in data]
    all_y_flat = [v for sub in all_y for v in sub] if all_y else []
    y_max = max(all_y_flat) if all_y_flat else 1
    
    N = 24*60*60
    for i in range(len(data)):
        x, y = data[i]
        data[i] = (x[:N], y[:N])

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
    }

    for (x, y), lab in zip(data, labels):
        style = style_map.get(lab, dict(color=None, linestyle='-', linewidth=1.6))
        plt.plot(
            x, y,
            label=lab,
            **style,
            solid_capstyle='round',
            solid_joinstyle='round'
        )

    # 坐标轴与标题
    plt.xlabel('time(hour)')
    plt.ylabel('codeblock coverage' if output_type == 'cov' else (output_type + ' num'))
    plt.title('Linux v6.2')

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
    parser = argparse.ArgumentParser(description="绘制日志覆盖率图或计算平均数据")
    parser.add_argument('--files', nargs='+', help="日志文件")
    parser.add_argument('--labels', nargs='+', help="标签")
    parser.add_argument('--type', help="类型")
    parser.add_argument('--out', default=None, help='输出图片路径（可选）')
    parser.add_argument('--average-out', default=None, help='输出平均数据文件路径（可选）')
    args = parser.parse_args()
    
    if args.type and args.type not in ["cov", "corpus", "exec"]:
        print("Invalid output type.")
        return
    
    # 如果指定了平均输出文件，则计算并保存平均数据
    if args.average_out and args.files and args.type:
        averaged_data = align_and_average_data(args.files, args.type)
        if averaged_data:
            save_averaged_data(averaged_data, args.average_out)
        else:
            print("无法计算平均数据")
    
    # 绘制图表
    if args.out or not args.average_out:
        # 合并所有要绘制的文件
        all_files = []
        if args.files:
            all_files.extend(args.files)
        
        # 准备标签
        labels = args.labels if args.labels else []
        
        # 绘制图表
        if all_files:
            plot_coverage(
                all_files=all_files,
                labels=labels,
                output_type=args.type if args.type else 'cov',
                out_path=args.out
            )

if __name__ == '__main__':
    main()