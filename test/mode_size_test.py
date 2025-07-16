# import torch
# from torch import nn
# from typing import List, Tuple

# # ===================================================================
# # 函数 1: 计算模型总大小
# # ===================================================================
# def get_model_size(model: nn.Module) -> float:
#     """计算并返回 PyTorch 模型的大小（单位：MB）。"""
#     state_dict = model.state_dict()
#     total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
#     size_in_mb = total_bytes / (1024 * 1024)
#     return round(size_in_mb, 2)

# # ===================================================================
# # 函数 2: 计算模型每一层的大小
# # ===================================================================
# def get_model_layer_size_list(model: nn.Module) -> List[Tuple[str, float]]:
#     """计算模型中每个层的大小，并返回一个包含层名称和大小的列表。"""
#     layer_sizes = []
#     for name, module in model.named_modules():
#         params_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
#         buffers_bytes = sum(b.numel() * b.element_size() for b in module.buffers(recurse=False))
#         total_bytes = params_bytes + buffers_bytes
#         if total_bytes > 0:
#             size_in_mb = total_bytes / (1024 * 1024)
#             layer_sizes.append((name, round(size_in_mb, 6)))
#     return layer_sizes

# # ===================================================================
# # 您定义的模型类
# # ===================================================================
# class SplitCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_features,
#                       32,
#                       kernel_size=5,
#                       padding=0,
#                       stride=1,
#                       bias=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32,
#                       64,
#                       kernel_size=5,
#                       padding=0,
#                       stride=1,
#                       bias=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2)),
#             nn.Flatten(start_dim=1) 
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(dim, 512), 
#             nn.ReLU(inplace=True)
#         )
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.fc1(out)
#         out = self.fc(out)
#         return out

# # ===================================================================
# #  主程序：实例化并使用函数
# # ===================================================================
# # 1. 实例化您的 SplitCNN 模型
# # 这里的 dim=1024 适用于输入图像为 28x28 的情况 (例如 MNIST)
# # (28x28 -> Conv1 -> 24x24 -> Pool1 -> 12x12 -> Conv2 -> 8x8 -> Pool2 -> 4x4.  64*4*4=1024)
# my_cnn_model = SplitCNN(in_features=1, num_classes=10, dim=1024)


# # 2. 使用 get_model_size 计算总大小
# total_model_size = get_model_size(my_cnn_model)
# print(f"模型 SplitCNN 的总大小为: {total_model_size} MB\n")


# # 3. 使用 get_model_layer_size_list 获取各层大小明细
# layer_info_list = get_model_layer_size_list(my_cnn_model)

# print("模型 SplitCNN 各层大小明细:")
# print("-" * 55)
# print(f"{'层名称 (Layer Name)':<35} | {'大小 (Size in MB)'}")
# print("-" * 55)

# calculated_total = 0
# for name, size in layer_info_list:
#     # 过滤掉空的容器层，只打印有实际大小的层
#     if size > 0:
#         print(f"{name:<35} | {size:>.6f}")
#         calculated_total += size

# print("-" * 55)
# print(f"{'明细加总 (Sum of Layers)':<35} | {round(calculated_total, 2)}")



# import torch
# from torch import nn
# from typing import Dict, Tuple
# import os

# # ===================================================================
# # 1. 定义你的模型 (这里使用之前的 SplitCNN)
# # ===================================================================
# class SplitCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_features, 32, kernel_size=5, bias=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=5, bias=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#         self.flatten_layer = nn.Flatten(start_dim=1)
#         self.fc1 = nn.Sequential(
#             nn.Linear(dim, 512),
#             nn.ReLU(inplace=True)
#         )
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.flatten_layer(out)
#         out = self.fc1(out)
#         out = self.fc(out)
#         return out

# # ===================================================================
# # 2. 定义用于存储激活值信息的字典和 Hook 函数
# # ===================================================================
# # 创建一个字典来保存每一层的激活值信息
# # 键是层的名称，值是一个元组 (形状, 内存大小MB)
# activation_info: Dict[str, Tuple[torch.Size, float]] = {}

# def get_activation_hook(name: str):
#     """
#     这是一个Hook的工厂函数，它返回一个真正的Hook函数。
#     使用工厂模式可以方便地将层的名称传入Hook中。
#     """
#     def hook(module, input, output):
#         """
#         Hook函数本身，会在每个模块的forward()之后被调用。
#         """
#         # 计算激活值占用的内存大小 (MB)
#         # output.numel() 是元素总数
#         # output.element_size() 是每个元素占用的字节数
#         memory_size_mb = (output.numel() * output.element_size()) / (1024 * 1024)
        
#         # 将形状和内存大小存入字典
#         activation_info[name] = (output.shape, round(memory_size_mb, 6))
        
#     return hook

# # ===================================================================
# # 3. 主程序：实例化模型，注册Hooks，并执行前向传播
# # ===================================================================
# # 实例化模型
# # 假设输入是 1x28x28 (如MNIST)，则dim=1024
# model = SplitCNN(in_features=1, num_classes=10, dim=1024)

# # 注册前向Hook到模型的每一层
# # model.named_modules() 会遍历模型中所有命名的模块
# for name, layer in model.named_modules():
#     layer.register_forward_hook(get_activation_hook(name))

# # 创建一个虚拟输入张量 (dummy input)
# # 形状: (batch_size, channels, height, width)
# # batch_size=64, channels=1, height=28, width=28
# dummy_input = torch.randn(64, 1, 28, 28)

# # 执行一次前向传播来触发所有的Hooks
# output = model(dummy_input)

# # ===================================================================
# # 4. 打印捕获到的激活值信息
# # ===================================================================
# print("模型各层激活值大小明细 (batch_size=64):")
# print("-" * 80)
# print(f"{'层名称 (Layer Name)':<35} | {'激活值形状 (Activation Shape)':<25} | {'内存大小 (MB)'}")
# print("-" * 80)

# for name, (shape, size_mb) in activation_info.items():
#     # 为了可读性，我们可以过滤掉没有输出的层或根模块
#     if shape:
#         print(f"{name:<35} | {str(shape):<25} | {size_mb:>.6f}")

# print("-" * 80)
# print(f"最终输出的形状: {output.shape}")




import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from collections import defaultdict

# 1. 您提供的自定义模型定义
class SplitCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(start_dim=1) 
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# 2. 实例化模型
#    为了分析，我们需要提供一个具体的输入尺寸，这里假设是28x28的灰度图（如MNIST）
#    根据模型结构，28x28的输入经过conv1和conv2后，展平的维度正好是1024
model = SplitCNN(in_features=1, num_classes=10, dim=1024)

# 3. 创建一个符合模型输入尺寸的伪输入
#    批量大小为1，通道为1，高和宽为28
input_tensor = torch.randn(1, 1, 28, 28)

# 4. 对整个模型进行一次完整的FLOPs分析
flop_analyzer = FlopCountAnalysis(model, input_tensor)
# 获取到每一个最底层模块的计算量
flops_by_module = flop_analyzer.by_module()

# 5. 将底层模块的计算量聚合到您定义的直接子模块中
#    (conv1, conv2, fc1, fc)
child_flops = defaultdict(float)
for name, flops in flops_by_module.items():
    # PyTorch的模块名是用'.'来分割层级的，例如'conv1.0'
    # 名字的第一部分就是我们定义的直接子模块的名称
    child_name = name.split('.')[0]
    if child_name in dict(model.named_children()):
        child_flops[child_name] += flops

# 6. 将聚合后的结果转换为您想要的列表元组格式
output_list = list(child_flops.items())

# 7. 打印最终结果
print("--- 每个自定义模块的计算量 ---")
print(output_list)

print("\n--- 格式化输出 ---")
for name, flops in output_list:
    print(f"模块 '{name}': {flops/1e6:.4f} MFLOPs")