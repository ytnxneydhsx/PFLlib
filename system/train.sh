#!/bin/bash

# 加载 Conda 初始化脚本
# 找到你的 Conda 根目录，例如：/home/huangnv_dl/miniconda3
# 然后将下面的路径替换成你自己的 Conda 安装路径
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活你的 Conda 环境
conda activate pfllib

# 检查环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "pfllib" ]; then
    echo "Error: Failed to activate Conda environment 'pfllib'."
    exit 1
fi

# 定义要运行的 Python 脚本
PYTHON_SCRIPT="main.py"

# 定义两个不同的配置名称
CONFIG1="SLCS_VGG16_Cifar10_alpha_0.8_data_Pruning_rate_0.9"
CONFIG2="SLCS_VGG16_Cifar10_alpha_0.8_data_Pruning_rate_0.8"

echo "Starting job with config: $CONFIG1"
# 使用 & 符号在后台运行第一个任务，并将输出重定向到日志文件
python "$PYTHON_SCRIPT" -pro "$CONFIG1" > job1.log 2>&1 &

# 暂停 1 秒
sleep 5

echo "Starting job with config: $CONFIG2"
# 使用 & 符号在后台运行第二个任务，并将输出重定向到另一个日志文件
python "$PYTHON_SCRIPT" -pro "$CONFIG2" > job2.log 2>&1 &

echo "All jobs started. Check job1.log and job2.log for output."

# 等待所有后台任务完成
wait

echo "All jobs finished."