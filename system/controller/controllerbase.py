import os
import torch
from torch import nn
import time
from typing import List, Tuple
from system.utils.data_utils import read_client_data
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis
from collections import defaultdict

class controllerbase():
    def __init__(self, args):
        self.device = args.device
        self.sample_data=self.get_sample_data(args)
        self.golbal_model_size=self.get_model_size(args.global_model)
        self.get_golbal_model_layer_size_list=self.get_model_layer_size_list(args.global_model)

        self.layer_activate_size_list= self.get_model_layer_activate_size_list(args.global_model, self.sample_data)
        self.layer_flops_list = self.get_layer_flops_list(args.global_model, self.sample_data)
        self.client_resource_list={}
        self.server_resource={}

    def get_model_size(self,model: nn.Module) -> float:

        state_dict = model.state_dict()
        total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())

        size_in_mb = total_bytes / (1024 * 1024)
        
        return size_in_mb
    def get_model_layer_size_list(self, model: nn.Module) -> list[tuple[str, float]]:
        layer_sizes_agg = defaultdict(float)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                params_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
                buffers_bytes = sum(b.numel() * b.element_size() for b in module.buffers(recurse=False))
                total_bytes = params_bytes + buffers_bytes
                if total_bytes > 0:
                    size_in_mb = total_bytes / (1024 * 1024)
                    child_name = name.split('.')[0]

                    layer_sizes_agg[child_name] += size_in_mb
                
        # 将聚合后的字典转换为列表
        output_list = []
        for name, size in layer_sizes_agg.items():
            output_list.append((name, round(size, 6)))
            
        return output_list
    
    def get_model_layer_activate_size_list(self,model: nn.Module,sample_data: torch.Tensor) -> List[Tuple[str, float]]:
            layer_activate_size_list = []
            handles = []
            for name, layer  in model.named_children():
                child_name = name.split('.')[0]
                handle = layer.register_forward_hook(
                self.get_activation_hook(child_name, layer_activate_size_list)
            )
                handles.append(handle)
            output = model(self.sample_data)
            for handle in handles:
                handle.remove()
            for name,size_mb in layer_activate_size_list:
                print(f"{name} | {size_mb}")
            return layer_activate_size_list
    
    def get_sample_data(self,args):

        data=read_client_data(args.dataset, 0)
        trainloader=DataLoader(data, args.batch_size, drop_last=True, shuffle=True)
        for i, (x, y) in enumerate(trainloader):
            sample_data = x
            if i== 0:
                break
        return sample_data.to(self.device)



    def get_activation_hook(self,name: str,layer_activate_size_list: List[Tuple[str, float]]):

        def hook(module, input, output):
            memory_size_mb = (output.numel() * output.element_size()) / (1024 * 1024)
            layer_activate_size_list.append((name,memory_size_mb))
        return hook
    
    def get_layer_flops_list(self, model: nn.Module, sample_data: torch.Tensor) -> List[Tuple[str, float]]:
        flop_analyzer = FlopCountAnalysis(model, sample_data)
        flops_by_module = flop_analyzer.by_module()
        child_flops = defaultdict(float)
        for name, flops in flops_by_module.items():
            child_name = name.split('.')[0]
            if child_name in dict(model.named_children()):
                child_flops[child_name] += flops/ (1024 * 1024)
        output_list = list(child_flops.items())
        return output_list







        








