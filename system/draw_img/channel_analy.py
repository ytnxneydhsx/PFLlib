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
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。请确保路径正确。")
        return None, None
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")
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
    # 使用正则表达式更精确地匹配，避免空块
    round_blocks = re.findall(r'-------------Round (\d+).*?Channel Kept Counts-------------.*?(\{[\s\S]*?\})', content, re.DOTALL)
    
    for round_num_str, counts_str in round_blocks:
        try:
            round_num = int(round_num_str)
            # 清理并解析字典字符串
            cleaned_counts_str = counts_str.replace('\n', '').replace(' ', '')
            counts_dict = eval(cleaned_counts_str)
            rounds_data[round_num] = counts_dict
        except (SyntaxError, NameError, ValueError) as e:
            print(f"警告：无法解析文件 {file_path} 中第 {round_num_str} 轮的字典。错误: {e}")
            continue
    
    if not rounds_data:
        print(f"警告：在 {file_path} 中未找到有效的轮次数据。")
        return params, None
        
    df = pd.DataFrame.from_dict(rounds_data, orient='index')
    df.sort_index(inplace=True)
    return params, df.transpose()

def generate_growth_heatmap_from_log(log_file_path, num_channels_to_plot=None):
    print(f"正在处理文件: {log_file_path}")
    
    # 1. 解析日志文件
    params, data_df = parse_log_file_full(log_file_path)
    
    if data_df is None or data_df.empty:
        print("无法生成图表，因为没有从日志文件中解析到数据。")
        return


    delta_df = data_df.diff(axis=1)
    

    ranked_df = delta_df.rank(method='dense', ascending=False)

    if num_channels_to_plot is not None and 0 < num_channels_to_plot < len(ranked_df):
        ranked_df = ranked_df.loc[0:num_channels_to_plot-1]
        print(f"--> 将只绘制前 {num_channels_to_plot} 个通道 (0 到 {num_channels_to_plot-1})。")

    base_name = f"{params.get('model', 'UnknownModel')}_{params.get('dataset', 'UnknownDataset')}"
    
    timestamp_str = "UnknownTime"
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', log_file_path)
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
    
    # 5.2 构建包含参数的基础字符串
    params_str = ""
    if params.get('alpha') is not None:
        params_str += f"_alpha:{params.get('alpha')}"
    if params.get('purning_min') is not None:
        params_str += f"_purning_min:{params.get('purning_min')}"
    if params.get('purning_base') is not None:
        params_str += f"_purning_base:{params.get('purning_base')}"
    if params.get('purning_max') is not None:
        params_str += f"_purning_max:{params.get('purning_max')}"
    if params.get('prune_tool') is not None:
        params_str += f"_prune_tool:{params.get('prune_tool')}"

    # 6. 创建并绘制热力图
    # 设置支持中文的字体，避免乱码警告
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 

    plt.style.use('seaborn-v0_8-whitegrid')
    height = max(6, num_channels_to_plot / 10 if num_channels_to_plot else 8)
    fig, ax = plt.subplots(figsize=(18, height))
    
    sns.heatmap(ranked_df, ax=ax, cmap='viridis_r', cbar=True, cbar_kws={'label': 'Rank of Count Increase (增长速率排名)'})
    
    # 6.1 【修改】构建包含时间戳的图表标题
    plot_title = f"{base_name}{params_str}"
    plot_title += f"\nTime: {timestamp_str}" # 在标题中加入时间
    plot_title += "\n(Color represents rank of growth from previous round)"
    if num_channels_to_plot:
        plot_title += f"\n(Displaying Top {num_channels_to_plot} Channels)"
        
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Training Round (训练轮次)', fontsize=12)
    ax.set_ylabel('Channel Index (信道索引)', fontsize=12)

    plt.tight_layout()
    
    # 7. 【修改】构建包含时间戳的输出文件名
    base_filename = f"{base_name}{params_str}"
    # 在文件名中加入时间戳
    output_filename = f"{base_filename}_time:{timestamp_str}_growth_rank"
    if num_channels_to_plot:
        output_filename += f"_top_{num_channels_to_plot}_channels"
    output_filename += '.jpg'
    # 替换掉文件名中可能存在的非法字符
    output_filename = output_filename.replace(":", "-")

    plt.savefig(output_filename)
    print(f"图表已成功保存为: {output_filename}\n")
    plt.close(fig)

# --- 使用方法 ---
# 在这里设置您想要分析的前n个通道数量
# 如果设置为 None，则会绘制所有通道
CHANNELS_TO_PLOT = 128

# 您的日志文件列表
files_to_process = [
    '/mnt/tjl/PFLlib/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/fixed_alpha/purning_min_0.6/purning_base_0.6/purning_max_0.6/lr_0.0003/split_model_cnt_2/0.0/time_2025-09-20_20-10-29/SLCDACP_VGG16_Cifar10_2025-09-20_20-10-29.log',
    '/mnt/tjl/PFLlib/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/fixed_alpha/purning_min_0.6/purning_base_0.6/purning_max_0.6/lr_0.0003/split_model_cnt_2/0.3/time_2025-09-19_14-37-45/SLCDACP_VGG16_Cifar10_2025-09-19_14-37-45.log',
    '/mnt/tjl/PFLlib/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/fixed_alpha/purning_min_0.6/purning_base_0.6/purning_max_0.6/lr_0.0003/split_model_cnt_2/0.6/time_2025-09-19_02-40-53/SLCDACP_VGG16_Cifar10_2025-09-19_02-40-53.log',
    '/mnt/tjl/PFLlib/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/fixed_alpha/purning_min_0.6/purning_base_0.6/purning_max_0.6/lr_0.0003/split_model_cnt_2/0.9/time_2025-09-19_03-43-18/SLCDACP_VGG16_Cifar10_2025-09-19_03-43-18.log'
     # Top-K
]

# 循环处理每个文件
for log_file in files_to_process:
    if os.path.exists(log_file):
        generate_growth_heatmap_from_log(log_file, num_channels_to_plot=CHANNELS_TO_PLOT)
    else:
        print(f"跳过，因为未在当前目录找到文件: {log_file}。请检查路径。")