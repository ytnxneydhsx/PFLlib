import torch
import os
import sys
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
import torch.nn as nn
from flcore.trainmodel.models import *
from torch.utils.data import DataLoader
import logging
import argparse
from typing import List, Union
from typing import List, Tuple
from utils.data_utils import read_client_data
from channelstools.channelstoolsbase import channelstoolsbase
from channelstools.channelstoolscdacp import channelstoolcdacp
import ast # 确保导入 ast 库
import configparser
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')


def model_load(model, model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def plot_group_loss_across_rounds(grouped_loss_data, train_round_list):
    num_groups = len(grouped_loss_data[0])
    group_labels = [f'Channel Group {i}' for i in range(num_groups)]
    if num_groups <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_groups))
    x = np.arange(len(train_round_list))
    total_width = 0.8
    bar_width = total_width / num_groups
    fig, ax = plt.subplots(figsize=(19.69, 13.90))
    for i in range(num_groups):
        data_for_group = [round_data[i] for round_data in grouped_loss_data]
        position = x + (i - (num_groups - 1) / 2) * bar_width
        ax.bar(position, data_for_group, bar_width, label=group_labels[i], color=colors[i])
    ax.set_ylabel('Loss Value', fontsize=40)
    ax.set_xlabel('Round', fontsize=40)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(train_round_list)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=20, loc='upper right')
    fig.tight_layout()
    plt.savefig("test.pdf")
    plt.close(fig)

def plot_single_round_loss(loss_values, round_label):
    num_groups = len(loss_values)
    group_labels = [f' {i}' for i in range(num_groups)]
    if num_groups <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_groups))
    fig, ax = plt.subplots(figsize=(19.69, 13.90))
    x_positions = np.arange(num_groups)
    for i in range(num_groups):
        ax.bar(x_positions[i], loss_values[i], color=colors[i], label=group_labels[i])
    ax.set_ylabel('Loss Value', fontsize=40)
    ax.set_xlabel('Channel Group', fontsize=40)
    # ax.set_title(f'Channel Group Loss for {round_label} round', fontsize=16)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # ax.legend(title='Channel Group', fontsize=20, title_fontsize=20)

    fig.tight_layout()
    plt.savefig("tes2.pdf")
    plt.close(fig) 



def config_load(args):
    if not args.profile:
        print("未指定 --profile，将仅使用命令行参数或代码中的默认值。")
        return
    print(f"检测到 profile: [{args.profile}]，将从 config.ini 加载并覆盖参数...")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')

    if args.profile not in config:
        print(f"错误: 在 config.ini 中未找到名为 '{args.profile}' 的配置区块!")
        sys.exit(1)


    for key, value in config.items(args.profile):
        if not hasattr(args, key):
            continue
        if value.lower() == 'none':
            converted_value = None
        else:
            try:
                converted_value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                converted_value = value
        setattr(args, key, converted_value)



def parser_initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pro', '--profile', type=str, default=None)
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-m', "--model", type=str, default="VGG16")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-smc', "--split_model_cnt", type=int, default=2)
    parser.add_argument('-cgn', "--channel_group_num", type=int, default=4)
    parser.add_argument('-mp', "--model_path", type=str, default=None)
    parser.add_argument('-bdci', "--bath_data_client_idx", type=int, default=None)
    parser.add_argument(
    '--train_round_list',  # 改为可选参数，使用--前缀
    '-trl',                # 可以添加一个简短的别名
    type=int,              # 指定每个元素的类型是整数
    nargs='+',             # 允许接收一个或多个值，并将它们放入一个列表中
    default=[10, 20, 30],  # 设置默认值为列表
    help='一个包含训练轮数的整数列表 (例如: --trl 5 15 25)'
)
    

    return parser
def convert_losses_to_percentages(loss_sum_list):
    percentage_loss_list = []
    
    # 遍历外层列表的每一个内部列表（对应每一轮round的结果）
    for inner_loss_list in loss_sum_list:
        # 1. 计算内部列表的总和
        total_loss = sum(inner_loss_list)
        
        # 2. 创建一个新的空列表来存放计算出的百分比
        inner_percentage_list = []
        
        # 处理特殊情况：如果总和为0，避免除以零的错误
        if total_loss == 0:
            # 如果所有损失都是0，那么每个损失的占比也是0
            inner_percentage_list = [0.0 for _ in inner_loss_list]
        else:
            # 3. 遍历内部列表的每一个损失值
            for loss in inner_loss_list:
                # 4. 计算百分比并添加到新列表中
                percentage = (loss / total_loss) * 100
                inner_percentage_list.append(percentage)
        
        # 5. 将计算好的内部百分比列表添加到最终结果中
        percentage_loss_list.append(inner_percentage_list)
        
    return percentage_loss_list








if __name__ == "__main__":
    parser=parser_initialize()
    args = parser.parse_args()
    config_load(args)
    args.model_str=args.model
    if args.model_str=="VGG16":
        if args.dataset == 'Cifar10':
            args.model=VGG16_cifar10().to('cuda')
    loss_sum_list=[]
    base_loss_list=[]
    for i in range(len(args.train_round_list)):
        model_path=args.model_path.replace("{round}",str(args.train_round_list[i]))
        model=model_load(args.model,model_path)
        channelstoolcdacp_test=channelstoolcdacp(args)
        channel_num=channelstoolcdacp_test.get_conv_layer_output_channels()
        channels_freeze_list_sum=channelstoolcdacp_test.get_channels_freeze_list_sum()
        free_loss_list=[]
        for i in range(args.channel_group_num):
            free_loss=channelstoolcdacp_test.analyze_layer_with_data(channels_freeze_list_sum[i])
            free_loss_list.append(free_loss)
        base_loss=channelstoolcdacp_test.get_base_loss()
        loss_sum_list.append(free_loss_list)
        base_loss_list.append(base_loss)

    percentage_loss_list=convert_losses_to_percentages(loss_sum_list)
    plot_group_loss_across_rounds(loss_sum_list,args.train_round_list)
    plot_single_round_loss(loss_sum_list[0],10)
    print(111111)

    
    


