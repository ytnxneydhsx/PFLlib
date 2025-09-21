import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data

class channelstoolsbase():
    def __init__(self,args):
        self.model=args.model
        self.split_model_cnt=args.split_model_cnt
        self.batch_size=args.batch_size
        self.dataset=args.dataset
        self.channel_num=self.get_conv_layer_output_channels()
        if args.bath_data_client_idx is not None:
           client_idx=args.bath_data_client_idx
           self.bath_data=self.bath_data_load(client_idx)
        else:
            self.bath_data=self.bath_data_load()

    def bath_data_load(self,client_idx=0):
        data = read_client_data(self.dataset, client_idx)
        data_loader=DataLoader(data, self.batch_size, drop_last=False, shuffle=True)
        for i, (x, y) in enumerate(data_loader):
            if type(x) == type([]):
                x[0] = x[0].to('cuda')
            else:
                x = x.to('cuda')
            y = y.to('cuda')
            if i==0:
                bath_data=(x,y)
                break
        return bath_data

    def get_conv_layer_output_channels(self):
        sequential_layers = list(self.model.children())
        if not 1 <= self.split_model_cnt <= len(sequential_layers):
            raise IndexError(
                f"层索引 {self.split_model_cnt} 超出范围。模型共有 {len(sequential_layers)} 个顺序执行的子模块。"
            )
        target_module = sequential_layers[self.split_model_cnt - 1]
        for layer in target_module.modules():
            if isinstance(layer, nn.Conv2d):
                return layer.out_channels
            elif isinstance(layer, nn.BatchNorm2d):
                return layer.num_features
        raise ValueError(
            f"在第 {self.split_model_cnt} 个模块中未能找到 nn.Conv2d 或 nn.BatchNorm2d 子层。"
            "无法确定其输出通道数。请确保查询的是卷积层。"
        )
    
    def get_base_loss(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval() 
        x, y =self.bath_data
        with torch.no_grad():
            output = self.model(x)
            loss = criterion(output, y)
        return loss.item()