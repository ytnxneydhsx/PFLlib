#!/usr/bin/env python

import sys
import os
import configparser
import matplotlib
matplotlib.use('Agg')
import copy
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import logging
from datetime import datetime
def setup_environment():

    # 简单的命令行参数解析，只为了获取 -pro 参数
    profile_name = None
    if '-pro' in sys.argv:
        try:
            # 获取 -pro 参数后面的值
            profile_name = sys.argv[sys.argv.index('-pro') + 1]
        except IndexError:
            # 如果没有值，则忽略
            pass

    if profile_name:
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')

        if profile_name in config:
            # 检查 device_id 是否存在于配置文件中
            if 'device_id' in config[profile_name]:
                device_id = config[profile_name]['device_id']
                print(f"检测到配置文件 [{profile_name}]，将设置 CUDA_VISIBLE_DEVICES={device_id}")
                # 核心步骤：在导入 torch 之前设置环境变量
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        else:
            print(f"警告: config.ini 中未找到配置区块 '{profile_name}'。将使用默认设备。")
    else:
        # 如果没有指定 -pro，则使用默认的 device_id=0
        if "--device_id" in sys.argv:
             try:
                device_id = sys.argv[sys.argv.index("--device_id") + 1]
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
                print(f"从命令行读取 device_id={device_id}，并设置环境变量。")
             except IndexError:
                 pass
        
        # 否则使用代码中的默认值（通常是 0）

# 立即调用这个函数
setup_environment()
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverfd import FD
from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverntd import FedNTD
from flcore.servers.servergh import FedGH
from flcore.servers.serverdbe import FedDBE
from flcore.servers.servercac import FedCAC
from flcore.servers.serverda import PFL_DA
from flcore.servers.serverlc import FedLC
from flcore.servers.serveras import FedAS
from flcore.servers.serversl import sl
from flcore.servers.serverfsl import fsl
from flcore.servers.serverslcs import slcs
from flcore.servers.serverslcdacp import slcdacp

from flcore.trainmodel.models import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

from data_select.datakmeans import  datakmeans
from data_select.datacenteragg import  datacenteragg
import random



def set_deterministic_seeds(seed=42):
    """设置所有随机种子以确保实验可复现性。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def run(args):
    """
    主训练和评估循环。
    """
    logger = logging.getLogger(__name__)
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        
        # 模型选择阶段
        if model_str == "MLR": 
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN":
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
        
        elif model_str == "split_CNN":
            if "MNIST" in args.dataset:
                 args.model = SplitCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)

        elif model_str=="ResNet18":
            if "MNIST" in args.dataset:
                args.model=ResNet18(ResBasicBlock,10).to(args.device)

        elif model_str == "DNN":
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                       num_classes=args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                      pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                      pool_kernel_size=(1, 2)).to(args.device)
        elif model_str == "VGG16":
            if args.dataset == 'Cifar10':
                args.model=VGG16_cifar10().to(args.device)
            if args.dataset =='MNIST':
                args.model=VGG_Simple_MNIST_Blocked().to(args.device)
        else:
            raise NotImplementedError
        
        args.dataset = build_full_dataset_path(args)
        print(args.model)
        logger.info(args.model)

        # 服务器选择阶段
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)

        elif args.algorithm == 'PFL-DA':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PFL_DA(args, i)

        elif args.algorithm == 'FedLC':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)

        elif args.algorithm == 'FedAS':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)
            
        elif args.algorithm == "SL":
            server = sl(args, i,2)
        
        elif args.algorithm == "FSL":
            server = fsl(args, i,2)

        elif args.algorithm == "SLCS":
            if args.data_select_name=='keames':
                args.data_select_obj=datakmeans
            elif args.data_select_name=='centeragg':
                args.data_select_obj=datacenteragg
            server = slcs(args, i,2)

        elif args.algorithm == "SLCDACP":
                server =slcdacp (args, i,args.split_model_cnt)
        else:
            raise NotImplementedError
        if args.staged_alpha_experiment=="start":
            server.train_staged_alpha_experiment()
        else:
            server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    logger.info(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    print("All done!")
    reporter.report()

def build_full_dataset_path(args):
    original_dataset_name = args.dataset
    base_dir = getattr(args, 'data_dir', 'MNIST_data') 
    
    full_path = os.path.join(
        base_dir,
        original_dataset_name,
        f"niid_{args.niid}",
        f"balance_{args.balance}",
        f"partition_{args.partition}",
        f"alpha_{args.alpha}",
        f"seed_{args.random_seed}",
        f"clients_{args.num_clients}"
    )
    return full_path

def apply_config_section(args_obj, config_obj, section_name, parser_obj):
    """将指定区块的配置应用到args对象上"""
    for key, value in config_obj.items(section_name):
        if hasattr(args_obj, key):
            try:
                arg_type = parser_obj._get_action_from_name(f"-{key.replace('_', '-')}")
                arg_type = arg_type.type if arg_type else str
            except (AttributeError, KeyError):
                default_val = parser_obj.get_default(key)
                arg_type = type(default_val) if default_val is not None else str

            try:
                if arg_type == bool:
                    converted_value = config_obj.getboolean(section_name, key)
                elif value.lower() == 'none':
                    converted_value = None
                else:
                    converted_value = arg_type(value)
                
                setattr(args_obj, key, converted_value)
            except (ValueError, TypeError) as e:
                print(f"警告：无法将INI中的值 '{value}' 转换为 {arg_type} 类型 (键: {key})。错误: {e}")
        else:
             print(f"警告：参数 '{key}' 在命令行参数中未定义。")


# 主执行函数
def main():
    """程序的主入口点，负责解析参数、配置环境和启动训练。"""
    
    parser = argparse.ArgumentParser()
    
    # 命令行参数定义
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False, help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False, help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="The threthold for droping slow clients")
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0, help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5, help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01, help="personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument('-M', "--M", type=int, default=5, help="Server only sends M client models to one client at each round")
    parser.add_argument('-itk', "--itk", type=int, default=4000, help="The iterations for solving quadratic subproblems")
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2, help="More fine-graind than its original paper.")
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
    parser.add_argument('-dsn',"--data_select_name",default=None)
    parser.add_argument('-pro', '--profile', type=str, default=None)
    parser.add_argument('-alpha', '--alpha', type=float, default=None)
    parser.add_argument('-niid', '--niid', type=bool, default=False)
    parser.add_argument('-balance', '--balance', type=bool, default=None)
    parser.add_argument('-partition', '--partition', type=str, default=None)
    parser.add_argument('-dsr', '--data_select_round', type=int, default=4000)
    parser.add_argument('-dpr', '--data_pruning_rate', type=float, default=0.8)
    parser.add_argument('-smc', '--split_model_cnt', type=int, default=2)
    parser.add_argument('-hln', '--hook_layer_name', type=str, default=None)
    parser.add_argument('-p_min', '--purning_min', type=float, default=0.1)
    parser.add_argument('-p_base', '--purning_base', type=float, default=0.2)
    parser.add_argument('-p_max', '--purning_max', type=float, default=0.3)
    parser.add_argument('-pt', '--prune_tool', type=str, default='default')
    parser.add_argument('-fa', '--fixed_alpha', type=float, default=0.2)
    parser.add_argument('-sae','--staged_alpha_experiment', type=str,default='default')
    parser.add_argument('-sl','--stage_length', type=int, default=20)
    parser.add_argument('-rs','--random_seed', type=int, default=1)
    parser.add_argument('-atv','--alpha_test_values', type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument('-ddir', '--data_dir', type=str, default='../dataset/')
    parser.add_argument('-ds', '--data_section', type=str, default=None)
    parser.add_argument('-op', '--optimizer_str', type=str, default=None)
    args = parser.parse_args()

    if args.profile:
        print(f"检测到 profile: [{args.profile}]，将从 config.ini 加载并覆盖参数...")
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')

        if args.profile not in config:
            print(f"错误: 在 config.ini 中未找到名为 '{args.profile}' 的配置区块!")
            sys.exit(1)

        if config.has_option(args.profile, 'data_section'):
            data_section_name = config.get(args.profile, 'data_section')
            if data_section_name in config:
                print(f"检测到 data_section，正在加载基础配置 [{data_section_name}]...")
                apply_config_section(args, config, data_section_name, parser)
            else:
                print(f"错误: config.ini 中未找到 data_section 指定的区块 '{data_section_name}'!")
                sys.exit(1)
        
        print(f"正在加载主 profile [{args.profile}] 的配置...")
        apply_config_section(args, config, args.profile, parser)

    else:
        print("未指定 --profile，将仅使用命令行参数或代码中的默认值。")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(args.data_dir)

    original_dataset_name = args.dataset
    
    
    print(f"数据集的完整路径被设置为: {args.dataset}")

    args.model_str = args.model
    args.current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_parent_dir = "logger"
    log_filename = f'{args.algorithm}_{args.model_str}_{original_dataset_name}_{args.current_date}.log'
    log_save_path = os.path.join(log_parent_dir, log_filename)
    logging.basicConfig(
        filename=log_save_path,
        filemode='a',
        level=logging.INFO,
        format='%(message)s'
    )
    logging.getLogger(__name__)

    print("=" * 50)
    print("最终生效的参数配置:")
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
        logging.info(f"{arg} = {getattr(args, arg)}")
    print("=" * 50)
    logging.info("=" * 50)
    
    set_deterministic_seeds(42)
    total_start = time.time()
    
    run(args)
    
    total_end = time.time()
    print(f"\nTotal time cost: {round(total_end - total_start, 2)}s.")
    
# 程序的执行入口
if __name__ == "__main__":
    main()