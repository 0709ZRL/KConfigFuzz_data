#!/bin/bash
# 4procs 1qemu 1cpu 4g

# sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/0919/kconfigfuzz/fuzz-1018.log kc
# sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/1104/kconfigfuzz/66-6p1q1c4g-1113.log kc-new 这个是新的，用这个
# sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/0919/syzkaller/fuzz-1017.log syz
# sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/0919/kconfigfuzz-ns/fuzz-1017.log kc-d
# sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/0919/kconfigfuzz-nd/fuzz-1017.log kc-v
# sshpass -p zhaoruilin0709 scp zzzrrll@10.30.6.1:~/workdir/log_6.6gcc_original healer
# sshpass -p zhaoruilin0709 scp zzzrrll@10.30.6.1:~/workdir_replace_SDM/log_6.6gcc_original healer-2
# sshpass -p hfl scp hfl@10.30.6.1:/home/hfl/hfl-release/scripts/syzkaller-4p1q1c4g-1024.log hfl-2

# 定义字符串数组（下标从 0 开始）
files=("kc-new" "syz" "kc-d" "kc-v" "healer" "hfl")
labels=("KConfigFuzz" "Syzkaller" "KConfigFuzz-D" "KConfigFuzz-V" "HEALER" "HFL")

# 输出提示信息
echo "可用选项（下标 → 内容）："
for i in "${!labels[@]}"; do
    echo "  $i → ${labels[i]}"
done

echo
echo -n "请输入你要输出的东西下标（用空格分隔，如 0 2 3）: "

# 读取用户输入的一整行
read -r input

# 检查是否输入为空
if [ -z "$input" ]; then
    echo "输入为空，退出。"
    exit 0
fi

# 临时保存输入为数组
indices=($input)  # split by whitespace

# 存放要传递给 draw.py 的文件参数（顺便把对应的标签也输入了）
file_args=()
label_args=()

# 遍历所有输入参数（这些都应该是数字）
for idx in "${indices[@]}"; do
    # 验证参数是否为数字
    if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        echo "错误: '$idx' 不是有效数字" >&2
        exit 1
    fi
    
    # 检查下标是否在数组范围内
    if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#files[@]}" ]; then
        echo "错误: 下标 $idx 超出范围 [0, $(( ${#files[@]} - 1 ))]" >&2
        exit 1
    fi
    
    # 添加对应字符串到参数列表
    file_args+=("${files[idx]}")
    label_args+=("${labels[idx]}")
    echo "添加了文件: ${files[idx]}, 标签: ${labels[idx]}"
done

echo
echo -n "请输入你要输出的类型：（1）cov 覆盖率 （2）corpus 产生新覆盖种子数 （3）exec 执行种子数: "
read -r type

echo
echo -n "请输入图片的保存路径："
read -r output_file

# 验证文件名是否为空
if [ -z "$output_file" ]; then
    echo "错误: 输出文件名不能为空。" >&2
    exit 1
fi

# 调试输出，查看最终要传递的参数
echo "最终文件参数: ${file_args[@]}"
echo "最终标签参数: ${label_args[@]}"
echo "类型: $type"
echo "输出文件: $output_file"

# 调用 draw_final.py，传入收集到的参数
if [ ${#file_args[@]} -eq 0 ]; then
    echo "没有有效参数，不执行 draw_final.py"
else
    # 正确构建参数
    python3 draw_final.py --files "${file_args[@]}" --labels "${label_args[@]}" --type "$type" --out "$output_file"
fi
