import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

# 设置全局随机种子
random.seed(1)
np.random.seed(1)
torch.manual_seed(1) # 增加 PyTorch 的随机种子设置

def generate_dataset(dir_path, num_clients, niid, balance, partition, alpha):
    """
    负责数据的下载、预处理、分割和保存。
    它会根据传入的参数来处理数据，并将处理结果保存在指定的 dir_path 中。
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 使用 os.path.join 来安全地拼接路径，兼容不同操作系统
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    rawdata_path = os.path.join(dir_path, "rawdata")

    # 检查数据是否已生成，如果已存在则直接返回
    if check(config_path, train_path, test_path, num_clients, alpha, niid, balance, partition):
        print(f"数据已在 {dir_path} 中生成，跳过。")
        return

    # 获取 FashionMNIST 数据
    print("正在下载和加载 FashionMNIST 数据集...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=rawdata_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=rawdata_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)
    
    # 从 DataLoader 中一次性获取所有数据
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    # 将数据和标签合并到 NumPy 数组中
    dataset_image = np.concatenate([
        trainset.data.cpu().detach().numpy(),
        testset.data.cpu().detach().numpy()
    ], axis=0)
    dataset_label = np.concatenate([
        trainset.targets.cpu().detach().numpy(),
        testset.targets.cpu().detach().numpy()
    ], axis=0)

    num_classes = len(np.unique(dataset_label)) # 使用 np.unique 更稳健
    print(f'类别总数: {num_classes}')

    # 调用数据分割和保存函数
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                      niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, alpha,
              statistic, niid, balance, partition)
    print("数据生成并保存完毕。")


def run_data_FashionMNIST_generation(config, section):
    
    print("\n--- [Module 1] 开始执行数据生成 (FashionMNIST) ---")
    dir_path = config.get(section, 'dir_path')
    num_clients = config.getint(section, 'num_clients')
    niid = config.getboolean(section, 'niid')
    balance = config.getboolean(section, 'balance')
    partition = config.get(section, 'partition')
    seed = config.getint(section, 'random_seed')
    alpha = config.getfloat(section, 'alpha')

    # 根据配置构建递归的文件夹结构
    full_dir_path = os.path.join(
        dir_path,
        f"niid_{niid}",
        f"balance_{balance}",
        f"partition_{partition}",
        f"alpha_{alpha}",
        f"seed_{seed}",
        f"clients_{num_clients}"
    )

    # 调用核心数据生成函数，传入新的完整路径
    generate_dataset(full_dir_path, num_clients, niid, balance, partition, alpha)