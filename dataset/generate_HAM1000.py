import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import random
import json
import configparser
from utils.dataset_utils import check, separate_data, split_data, save_file
# 假设您的 utils.dataset_utils 文件包含这些函数
# from utils.dataset_utils import check, separate_data, split_data





def generate_ham1000_dataset(output_dir, source_data_path, num_clients, niid, balance, partition, alpha):
    """
    负责HAM1000数据的加载、预处理、分割和保存。
    - output_dir: 处理后数据的保存路径 (例如: ./data/HAM1000/niid_True/...)
    - source_data_path: 原始数据所在的路径 (例如: ./data/HAM1000/rawdata)
    """
    config_path = os.path.join(output_dir, "config.json")
    train_path = os.path.join(output_dir, "train/")
    test_path = os.path.join(output_dir, "test/")

    if check(config_path, train_path, test_path, num_clients, alpha, niid, balance, partition):
        print(f"数据已在 {output_dir} 中生成，跳过。")
        return

    print(f"正在从 {source_data_path} 加载和处理 HAM1000 数据集...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    metadata_path = os.path.join(source_data_path, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    df['dx_id'] = pd.Categorical(df['dx']).codes
    num_classes = len(df['dx_id'].unique())
    print(f'类别总数: {num_classes}')

    dataset_image = []
    dataset_label = []
    image_folder_1 = os.path.join(source_data_path, 'HAM10000_images_part_1')
    image_folder_2 = os.path.join(source_data_path, 'HAM10000_images_part_2')

    print("开始从文件夹加载图像...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_path = os.path.join(image_folder_1, f"{row['image_id']}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_folder_2, f"{row['image_id']}.jpg")
        
        if not os.path.exists(image_path):
            print(f"警告: 找不到图像 {row['image_id']}.jpg，跳过。")
            continue
            
        with Image.open(image_path).convert('RGB') as img:
            dataset_image.append(transform(img).cpu().detach().numpy())
            dataset_label.append(row['dx_id'])

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    print(f"数据加载完毕。图像形状: {dataset_image.shape}, 标签形状: {dataset_label.shape}")

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, alpha,
              statistic, niid, balance, partition)

def run_data_HAM1000_generation(config, section):
    print("\n--- [Module 1] 开始执行 HAM1000 数据生成 ---")
    
    # 1. 从配置文件中读取所有参数
    # **修改**: 分离源路径和输出基础路径
    source_data_path = config.get(section, 'source_data_path')
    output_base_path = config.get(section, 'output_base_path')
    num_clients = config.getint(section, 'num_clients')
    niid = config.getboolean(section, 'niid')
    balance = config.getboolean(section, 'balance')
    partition = config.get(section, 'partition')
    seed = config.getint(section, 'random_seed')
    alpha = config.getfloat(section, 'alpha')

    # 2. **修改**: 像Cifar10脚本一样，在 output_base_path 下构建层次化的完整输出路径
    full_output_path = os.path.join(
        output_base_path,
        f"niid_{niid}",
        f"balance_{balance}",
        f"partition_{partition}",
        f"alpha_{alpha}",
        f"seed_{seed}",
        f"clients_{num_clients}"
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 3. **修改**: 调用核心函数，传入新的完整输出路径和独立的源数据路径
    generate_ham1000_dataset(full_output_path, source_data_path, num_clients, niid, balance, partition, alpha)
    print("--- [Module 1] HAM1000 数据生成完毕 ---")


# if __name__ == '__main__':
#     config = configparser.ConfigParser()
#     config.add_section('HAM1000_SETUP')

#     # **修改**: 设置分离的路径，与 Cifar10 脚本逻辑一致
#     # source_data_path: 指向包含原始数据（CSV和图片文件夹）的目录
#     # output_base_path: 用于存放所有生成的数据集的根目录
#     config.set('HAM1000_SETUP', 'source_data_path', './data/HAM1000/rawdata') 
#     config.set('HAM1000_SETUP', 'output_base_path', './data/HAM1000')      

#     config.set('HAM1000_SETUP', 'num_clients', '10')
#     config.set('HAM1000_SETUP', 'niid', 'True')
#     config.set('HAM1000_SETUP', 'balance', 'False')
#     config.set('HAM1000_SETUP', 'partition', 'dir')
#     config.set('HAM1000_SETUP', 'alpha', '0.5')
#     config.set('HAM1000_SETUP', 'random_seed', '42')

#     run_data_HAM1000_generation(config, 'HAM1000_SETUP')