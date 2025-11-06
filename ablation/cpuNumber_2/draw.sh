#!/bin/bash

# 测试6proc 1qemu 4G下不同的CPU数对测试效率的影响
# 理论上只控制CPU数对测试效率没什么影响，因为如果你无法充分利用测试资源，即调整proc，那对测试效率没什么显著提升。
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/kconfigfuzz/66-6p1q8c4g-1104.log kc-nod-8c
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/kconfigfuzz/66-6p1q4c4g-1104.log kc-nod-4c

# 6p1q2c4g下将动态验证放到Minimize前无验证队列的KC
sshpass -p jiakai scp jiakai@10.30.6.1:/home/jiakai/vini/kconfigfuzz/66-6p1q2c4g-1104.log kc-nod-2c

# 定义字符串数组（下标从 0 开始）
files=("kc" "kc-nod-2c" "kc-nod-4c" "kc-nod-8c")
labels=("KC-1CPU" "KC-2CPU(current)" "KC-4CPU" "KC-8CPU")

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