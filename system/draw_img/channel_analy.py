import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file_full(file_path):
    """
    解析指定的日志文件，提取头部参数和每一轮的 'Channel Kept Counts'。
    返回一个包含参数的字典和一个包含计数的DataFrame。
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。请确保路径正确。")
        return None, None

    # 1. 解析头部参数
    params = {}
    header = content.split('==================================================')[0]
    param_keys = ['model', 'dataset', 'alpha', 'purning_min', 'purning_base', 'purning_max', 'prune_tool']
    for line in header.split('\n'):
        parts = line.split('=')
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            if key in param_keys:
                params[key] = value
    
    # 2. 解析 Channel Kept Counts
    rounds_data = {}
    rounds_raw = content.split('-------------Round ')
    
    for round_block in rounds_raw[1:]:
        round_match = re.match(r'(\d+).*?Channel Kept Counts-------------', round_block, re.DOTALL)
        if round_match:
            round_num = int(round_match.group(1))
            
            counts_str_match = re.search(r'(\{[\s\S]*?\})', round_block)
            if counts_str_match:
                try:
                    counts_str = counts_str_match.group(1).replace('\n', '').replace(' ', '')
                    counts_dict = eval(counts_str)
                    rounds_data[round_num] = counts_dict
                except (SyntaxError, NameError) as e:
                    print(f"警告：无法解析文件 {file_path} 中第 {round_num} 轮的字典。错误: {e}")
                    continue
    
    if not rounds_data:
        print(f"警告：在 {file_path} 中未找到有效的轮次数据。")
        return params, None
        
    df = pd.DataFrame.from_dict(rounds_data, orient='index')
    df.sort_index(inplace=True)
    return params, df.transpose()

def generate_growth_heatmap_from_log(log_file_path, num_channels_to_plot=None):
    """
    接收一个日志文件的绝对路径，生成信道重要性【增长速率】的热力图，
    并以.jpg格式保存在当前工作目录。
    
    :param log_file_path: 日志文件的路径。
    :param num_channels_to_plot: (可选) 要绘制的前n个通道的数量。如果为None，则绘制所有通道。
    """
    print(f"正在处理文件: {log_file_path}")
    
    # 1. 解析日志文件
    params, data_df = parse_log_file_full(log_file_path)
    
    if data_df is None or data_df.empty:
        print("无法生成图表，因为没有从日志文件中解析到数据。")
        return

    # 2. 【核心改动】计算每轮的增量
    # .diff(axis=1) 计算当前列与前一列的差值
    delta_df = data_df.diff(axis=1)
    
    # 3. 对增量数据进行排名处理
    ranked_df = delta_df.rank(method='dense', ascending=False)

    # 4. 如果指定了通道数，则筛选数据
    if num_channels_to_plot is not None and 0 < num_channels_to_plot < len(ranked_df):
        ranked_df = ranked_df.loc[0:num_channels_to_plot-1]
        print(f"--> 将只绘制前 {num_channels_to_plot} 个通道 (0 到 {num_channels_to_plot-1})。")

    # 5. 根据规则构建标题和文件名
    base_name = f"{params.get('model', 'UnknownModel')}_{params.get('dataset', 'UnknownDataset')}"
    title_str = base_name
    
    if params.get('alpha') is not None:
        title_str += f"_alpha:{params.get('alpha')}"
    if params.get('purning_min') is not None:
        title_str += f"_purning_min:{params.get('purning_min')}"
    if params.get('purning_base') is not None:
        title_str += f"_purning_base:{params.get('purning_base')}"
    if params.get('purning_max') is not None:
        title_str += f"_purning_max:{params.get('purning_max')}"
    if params.get('prune_tool') is not None:
        title_str += f"_prune_tool:{params.get('prune_tool')}"

    # 6. 创建并绘制热力图
    plt.style.use('seaborn-v0_8-whitegrid')
    height = max(6, num_channels_to_plot / 10 if num_channels_to_plot else 8)
    fig, ax = plt.subplots(figsize=(18, height))
    
    sns.heatmap(ranked_df, ax=ax, cmap='viridis_r', cbar=True, cbar_kws={'label': 'Rank of Count Increase (增长速率排名)'})
    
    plot_title = title_str + "\n(Color represents rank of growth from previous round)"
    if num_channels_to_plot:
        plot_title += f"\n(Displaying Top {num_channels_to_plot} Channels)"
        
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Training Round (训练轮次)', fontsize=12)
    ax.set_ylabel('Channel Index (信道索引)', fontsize=12)

    plt.tight_layout()
    
    # 7. 保存图像
    output_filename = title_str + "_growth_rank" # 添加后缀以区分
    if num_channels_to_plot:
        output_filename += f"_top_{num_channels_to_plot}_channels"
    output_filename += '.jpg'

    plt.savefig(output_filename)
    print(f"图表已成功保存为: {output_filename}\n")
    plt.close(fig)

# --- 使用方法 ---
# 在这里设置您想要分析的前n个通道数量
# 如果设置为 None，则会绘制所有通道
CHANNELS_TO_PLOT = 100

# 您的日志文件列表
files_to_process = [
'/home/huangnv_dl/PFLlib-master/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/purning_max_bath_top-k/purning_min_0.1/purning_base_0.2/purning_max_0.3/time_2025-09-12_10-56-07/SLCDACP_VGG16_Cifar10_2025-09-12_10-56-07.log', # Top-K
'/home/huangnv_dl/PFLlib-master/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/purning_max_defult/purning_min_0.1/purning_base_0.2/purning_max_0.3/time_2025-09-12_10-55-57/SLCDACP_VGG16_Cifar10_2025-09-12_10-55-57.log', # CDACP ('defult')
'/home/huangnv_dl/PFLlib-master/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/purning_max_random/purning_min_0.1/purning_base_0.2/purning_max_0.3/time_2025-09-12_10-56-03/SLCDACP_VGG16_Cifar10_2025-09-12_10-56-03.log'  # Random
]

# 循环处理每个文件
for log_file in files_to_process:
    if os.path.exists(log_file):
        generate_growth_heatmap_from_log(log_file, num_channels_to_plot=CHANNELS_TO_PLOT)
    else:
        print(f"跳过，因为未在当前目录找到文件: {log_file}。请检查路径。")


