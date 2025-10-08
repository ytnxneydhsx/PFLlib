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
    "49b3db2c8df80737266ad1bae48405df6f3a2519"
    "4d5f101f1eef4921a30110cee9ac8a10a2fcd31f"
    "4ec88b017ccfb4a50ea050f9b1c591345b2a1729"
    "35533d78667ec01f9d5dad6e1466974c06877486"
)

# 每批次运行的任务数量
BATCH_SIZE=4

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