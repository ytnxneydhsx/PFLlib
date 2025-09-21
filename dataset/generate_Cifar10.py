import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
# 请确保这个文件存在并且包含所需的函数
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Cifar10/"


def generate_dataset(dir_path, num_clients, niid, balance, partition, alpha):
    """
    负责数据的下载、预处理、分割和保存。
    它会根据传入的参数（如NIID、平衡性、分区等）来处理数据。
    """
    # 确保文件夹存在
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 设置训练/测试数据的目录
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    # 检查数据是否已经生成，如果已存在则直接返回
    if check(config_path, train_path, test_path, num_clients, alpha, niid, balance, partition):
        print(f"数据已在 {dir_path} 中生成，跳过。")
        return

    # 获取 Cifar10 数据集
    print("正在下载和加载 CIFAR10 数据集...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=os.path.join(dir_path, "rawdata"), train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(dir_path, "rawdata"), train=False, download=True, transform=transform)
    
    # 将整个数据集加载到内存
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)
    
    train_data_batch = next(iter(trainloader))
    test_data_batch = next(iter(testloader))
    
    dataset_image = np.concatenate([
        train_data_batch[0].cpu().detach().numpy(),
        test_data_batch[0].cpu().detach().numpy()
    ], axis=0)
    
    dataset_label = np.concatenate([
        train_data_batch[1].cpu().detach().numpy(),
        test_data_batch[1].cpu().detach().numpy()
    ], axis=0)

    num_classes = len(set(dataset_label))
    print(f'类别总数: {num_classes}')

    # 调用数据分割和保存函数
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                     niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, alpha,
              statistic, niid, balance, partition)
    print("数据生成并保存完毕。")


def run_data_Cifar10_generation(config, section):

    print("\n--- [Module 1] 开始执行数据生成 ---")

    # 从配置文件中读取所有参数
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

    # 使用配置来设置环境，例如随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 调用核心数据生成函数，传入新的完整路径
    generate_dataset(full_dir_path, num_clients, niid, balance, partition, alpha)



