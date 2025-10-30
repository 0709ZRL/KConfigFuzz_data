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


def plot_coverage(log_paths: List[str], labels: List[str], output_type: str, out_path: Optional[str] = None):
    """
    绘制曲线并输出图像。
    log_paths: 所有日志文件路径
    labels: 所有曲线的标签
    output_type: 输出的类型
    out_path: 保存图片的路径，若为 None 则显示图形
    """

    # 读取三份日志
    data = []
    for i, path in enumerate(log_paths):
        entries = read_log(path, output_type)
        x, y, _t0 = to_elapsed_hours(entries)
        data.append((x, y))

    # 计算图形上的最大 y 值，以便设置合适的 y 轴范围（可选）
    all_y = [yv for _, yv in data]
    # Flatten
    all_y_flat = [v for sub in all_y for v in sub] if all_y else []
    y_max = max(all_y_flat) if all_y_flat else 1

    N = 48*60*6
    # N = len(data[0][0])
    # for i in range(len(data)):
    #     if len(data[i][0]) < N:
    #         N = len(data[i][0])

    # 画图
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        x, y = data[i]
        plt.plot(x[:N], y[:N], label=labels[i], linewidth=2)

    plt.xlabel('Time(hours)')
    plt.ylabel('Codeblock Coverage')
    #plt.ylabel(output_type + ' num')
    plt.title('Linux v6.6')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=max(1.0, y_max * 1.05))

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