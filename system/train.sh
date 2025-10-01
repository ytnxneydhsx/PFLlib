#!/bin/bash

# --- Self-locating and setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STATUS_FILE="$SCRIPT_DIR/tmp/training_status.log"
mkdir -p "$SCRIPT_DIR/tmp"
# ------------------------------

# 获取 Conda 根目录，并加载初始化脚本
# 根据你的 'which conda' 输出，你的 Conda 根目录是 /home/yons/anaconda3
source /home/yons/anaconda3/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate tjl_fl

# 检查环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "tjl_fl" ]; then
    echo "Error: Failed to activate Conda environment 'tjl_fl'."
    exit 1
fi

# 定义要运行的 Python 脚本
PYTHON_SCRIPT="main.py"

# 定义所有配置名称
CONFIGS=(
    "65d31920b99ccb9028b4051cb5a9b76854af6a07"
    "02104122fe8afed115f2a9741a85ba62ca95c9d9"
    "76fb2e0cdfa5dc0e3ceb4d4aede3376213c71334"
    "77dd0927a7f182217d9eb2a2f78a09b818e0f082"
)

# 每批次运行的任务数量
BATCH_SIZE=8

for ((i=0; i<${#CONFIGS[@]}; i+=BATCH_SIZE))
do
    echo "--- Starting a new batch of jobs ---"

    # 启动当前批次的任务
    for ((j=0; j<BATCH_SIZE; j++))
    do
        # 检查数组索引是否越界
        if ((i+j < ${#CONFIGS[@]}))
        then
            config_name=${CONFIGS[i+j]}
            log_file="job_${config_name}.log"
            
            echo "Starting job with config: $config_name"
            # Write current config to status file for the UI to read
            echo "$config_name" > "$STATUS_FILE"
            python "$PYTHON_SCRIPT" -pro "$config_name" > "$log_file" 2>&1 &

            # 在启动下一个任务前，等待5秒
            sleep 5
        fi
    done
    
    echo "All jobs in the current batch started. Waiting for them to finish..."
    
    # 等待当前批次的所有后台任务完成
    wait

    echo "All jobs in the current batch finished."
done

echo "--- All jobs finished ---"
# Clear the status file to indicate completion
> "$STATUS_FILE"