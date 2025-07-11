import torch
import torch.nn as nn
from collections import OrderedDict

# --- FedAvgCNN 模型定义 ---
class FedAvgCNN(nn.Module):
    """
    改造后的 FedAvgCNN 模型，将 flatten 操作封装为 nn.Flatten 模块，
    以便模型更具普适性，能够被 split_model 函数正确分割。
    """
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # 将 flatten 操作封装为 nn.Flatten 模块
        # start_dim=1 表示从批次维度（第0维）之后开始展平
        self.flatten = nn.Flatten(start_dim=1) 
        
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flatten(out) 
        out = self.fc1(out)
        out = self.fc(out)
        return out

# --- split_model 函数定义 ---
def split_model(model: nn.Module, split_point: int) -> tuple[nn.Module, nn.Module]:
    """
    将给定的 nn.Module 模型切分为两个 nn.Sequential 模块。

    参数:
        model (nn.Module): 要切分的模型。
        split_point (int): 分割点，表示在模型的第 `split_point` 层之后进行切分。
                            （从0开始计数，split_point 层包含在第一个返回的模型中）。

    返回:
        tuple[nn.Module, nn.Module]: 一个包含两个 nn.Sequential 模块的元组 (model_part1, model_part2)。
                                      如果 split_point 超出范围，将抛出 ValueError 或 IndexError。
    """
    all_layers = OrderedDict(model.named_children())
    
    if not all_layers:
        raise ValueError("模型没有可切分的子层。")
    # split_point 应该在 [0, len(all_layers) - 1] 范围内
    if split_point < 0 or split_point >= len(all_layers): 
        raise IndexError(f"切分点 {split_point} 超出模型层数范围 ({len(all_layers)} 层)。有效范围是 [0, {len(all_layers)-1}]。")

    layers_part1 = OrderedDict()
    layers_part2 = OrderedDict()

    for i, (name, layer) in enumerate(all_layers.items()):
        if i <= split_point: # 包含 split_point 对应的层
            layers_part1[name] = layer
        else:
            layers_part2[name] = layer

    model_part1 = nn.Sequential(layers_part1)
    model_part2 = nn.Sequential(layers_part2)

    return model_part1, model_part2

# --- 测试用例和示例用法 ---
if __name__ == "__main__":
    # 1. 实例化原始模型
    # 假设输入是 28x28 的灰度图像，因此 dim=1024 (64 * 4 * 4) 是正确的
    original_model = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
    print("--- 原始模型结构 ---")
    print(original_model)
    print(f"原始模型层数: {len(list(original_model.named_children()))}\n")

    # 2. 定义一个模拟输入数据
    # batch_size=1，单通道，28x28图像
    dummy_input = torch.randn(1, 1, 28, 28)
    print(f"模拟输入数据形状: {dummy_input.shape}\n")

    # --- 测试切分点 0 (在 conv1 之后) ---
    print("=== 测试切分点 0 (conv1 之后) ===")
    split_point_idx_0 = 0
    head_model_0, tail_model_0 = split_model(original_model, split_point=split_point_idx_0)

    print("\n模型头部 (split_point_idx=0):")
    print(head_model_0) # 包含 conv1
    print("\n模型尾部 (split_point_idx=0):")
    print(tail_model_0) # 包含 conv2, fc1, fc

    # 运行切分后的模型
    print("\n--- 运行切分后的模型 (split_point_idx=0) ---")
    output_head_0 = head_model_0(dummy_input)
    print(f"头部模型输出形状: {output_head_0.shape}") # 应该是 [1, 32, 12, 12]

    # 注意：这里 tail_model_0 的第一个是 conv2，所以输入仍然是特征图
    # 在这个切分点，不需要额外的手动 flatten，因为 tail_model_0 的第一个层 (conv2) 接受的正是这种形状
    output_combined_0 = tail_model_0(output_head_0)
    print(f"组合模型输出形状: {output_combined_0.shape}") # 应该是 [1, 10] (最终分类输出)

    # 验证与原始模型输出是否一致
    original_output = original_model(dummy_input)
    print(f"原始模型输出形状: {original_output.shape}")
    print(f"切分点0的输出与原始模型输出是否近似一致: {torch.allclose(output_combined_0, original_output)}\n")

    # --- 测试切分点 1 (在 conv2 之后) ---
    print("=== 测试切分点 1 (conv2 之后) ===")
    split_point_idx_1 = 1
    head_model_1, tail_model_1 = split_model(original_model, split_point=split_point_idx_1)

    print("\n模型头部 (split_point_idx=1):")
    print(head_model_1) # 包含 conv1, conv2
    print("\n模型尾部 (split_point_idx=1):")
    print(tail_model_1) # 包含 fc1, fc

    # 运行切分后的模型
    print("\n--- 运行切分后的模型 (split_point_idx=1) ---")
    output_head_1 = head_model_1(dummy_input)
    print(f"头部模型输出形状: {output_head_1.shape}") # 应该是 [1, 64, 4, 4]

    # !! 关键步骤：在传递给尾部模型之前，手动执行 flatten 操作 !!
    flattened_output_1 = torch.flatten(output_head_1, 1)
    print(f"展平后的特征形状: {flattened_output_1.shape}") # 应该是 [1, 1024]

    output_combined_1 = tail_model_1(flattened_output_1)
    print(f"组合模型输出形状: {output_combined_1.shape}") # 应该是 [1, 10]

    # 验证与原始模型输出是否一致
    print(f"切分点1的输出与原始模型输出是否近似一致: {torch.allclose(output_combined_1, original_output)}\n")

    # --- 测试切分点 2 (在 fc1 之后) ---
    print("=== 测试切分点 2 (fc1 之后) ===")
    split_point_idx_2 = 2
    head_model_2, tail_model_2 = split_model(original_model, split_point=split_point_idx_2)

    print("\n模型头部 (split_point_idx=2):")
    print(head_model_2) # 包含 conv1, conv2, fc1
    print("\n模型尾部 (split_point_idx=2):")
    print(tail_model_2) # 包含 fc

    # 运行切分后的模型
    print("\n--- 运行切分后的模型 (split_point_idx=2) ---")
    # 由于 head_model_2 内部包含了 conv1, conv2 和 fc1，并且 fc1 前面原始模型有 flatten，
    # nn.Sequential 在处理时会按顺序调用这些子模块的 forward 方法。
    # 原始模型的 forward 是 conv1 -> conv2 -> flatten -> fc1 -> fc
    # 这里的 head_model_2 包含 conv1, conv2, fc1，但它本身没有 flatten 层。
    # 这意味着我们需要手动处理 flatten，或者 head_model_2 的 forward 方法需要被重写来包含它。
    # 
    # 更安全的做法是：
    # 1. 直接通过 head_model_2 (它内部会处理 conv1 -> conv2 -> flatten -> fc1 的逻辑)
    output_head_2 = head_model_2(dummy_input) 
    print(f"头部模型输出形状: {output_head_2.shape}") # 应该是 [1, 512]

    # 2. 将头部输出传递给尾部模型
    output_combined_2 = tail_model_2(output_head_2)
    print(f"组合模型输出形状: {output_combined_2.shape}") # 应该是 [1, 10]

    # 验证与原始模型输出是否一致
    print(f"切分点2的输出与原始模型输出是否近似一致: {torch.allclose(output_combined_2, original_output)}\n")

    # --- 测试无效切分点 ---
    print("=== 测试无效切分点 ===")
    try:
        # 尝试切分点超出最大索引
        split_model(original_model, split_point=len(list(original_model.named_children())))
    except IndexError as e:
        print(f"成功捕获到预期错误: {e}")

    try:
        # 尝试切分点为负数
        split_model(original_model, split_point=-1)
    except IndexError as e:
        print(f"成功捕获到预期错误: {e}")