import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Cifar10/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition,alpha):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients,alpha, niid, balance, partition):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, alpha,
        statistic, niid, balance, partition)




def run_data_Cifar10_generation(config):

    print("\n--- [Module 1] 开始执行数据生成 ---")
    # random.seed(1)
    # np.random.seed(1)
    # num_clients = 5
    # dir_path = "Cifar10/"

    # 1. 从传入的 config 对象中读取所有配置
    #    使用 .getint(), .getboolean() 来自动转换类型
    section = 'Cifar10_DATA_GENERATION'
    dir_path = config.get(section, 'dir_path')
    num_clients = config.getint(section, 'num_clients')
    niid = config.getboolean(section, 'niid')
    balance = config.getboolean(section, 'balance')
    partition = config.get(section, 'partition')
    seed = config.getint(section, 'random_seed')
    alpha= config.getfloat(section, 'alpha')

    # 2. 使用配置来设置环境，例如随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 3. 调用核心工作函数，传入从配置中读取的值
    generate_dataset(dir_path, num_clients, niid, balance, partition,alpha)


# if __name__ == "__main__":
#     # niid = True if sys.argv[1] == "noniid" else False
#     # balance = True if sys.argv[2] == "balance" else False
#     # partition = sys.argv[3] if sys.argv[3] != "-" else None

#     # generate_dataset(dir_path, num_clients, niid, balance, partition)
#     generate_dataset(dir_path, 5, False, True, None,None)