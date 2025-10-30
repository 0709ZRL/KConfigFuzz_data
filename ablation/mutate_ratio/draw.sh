#!/bin/bash
# 6procs 2cpu 4g
# 在同一版本上跑kc
# 默认变异策略：我们的编译策略的概率比
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/mutate/kconfigfuzz2-1/66-6p1q2c4g-1029-21.log kc-2-1
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/mutate/kconfigfuzz1-1/66-6p1q2c4g-1029-11.log kc-1-1
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/kconfigfuzz/vini-66-6p1q2c4g-1029.log kc-1-2 #(当前)
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/mutate/kconfigfuzz1-4/66-6p1q2c4g-1029-14.log kc-1-4
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/mutate/kconfigfuzz1-8/66-6p1q2c4g-1029-18.log kc-1-8

# 定义字符串数组（下标从 0 开始）
files=("kc-2-1" "kc-1-1" "kc-1-2" "kc-1-4" "kc-1-8")
labels=("KC-2-1" "KC-1-1" "KC-1-2(current)" "KC-1-4" "KC-1-8")

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