# import torch
# import torch.nn as nn
# from typing import List, Union
# from typing import List, Tuple
# from channelstools.channelstoolsbase import channelstoolsbase

# class channelstoolcdacp(channelstoolsbase):
#     def __init__(self, args):
#         super().__init__(args)
#         self.channel_group_num=args.channel_group_num


#     def get_channels_freeze_list_sum(self):
#         group_size = self.channel_num // self.channel_group_num
#         channels_freeze_list_sum = []
#         for i in range(self.channel_group_num):
#             start_channel = i * group_size
#             if i == self.channel_group_num - 1:
#                 end_channel = self.channel_num
#             else:
#                 end_channel = (i + 1) * group_size
#             channels_freeze_list = list(range(start_channel, end_channel))
#             channels_freeze_list_sum.append(channels_freeze_list)
#         return channels_freeze_list_sum
    
#     def analyze_layer_with_data(self,channels_freeze_list):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.eval()
#         self.model.to(device)
#         inputs, labels = self.bath_data
#         inputs, labels = inputs.to(device), labels.to(device)
#         sequential_layers = list(self.model.children())
#         if not 1 <= self.split_model_cnt <= len(sequential_layers):
#             raise IndexError(
#                 f"Layer index {self.split_model_cnt} is out of bounds. Model has {len(sequential_layers)} sequential modules."
#             )
#         target_layer = sequential_layers[self.split_model_cnt - 1]
#         criterion = nn.CrossEntropyLoss()
#         hook_handle = None
#         if channels_freeze_list:
#             def create_freezer_hook(channels):
#                 def hook(module, input, output):
#                     if output.dim() == 4:  
#                         output[:, channels, :, :] = 0.0
#                     elif output.dim() == 2:  
#                         output[:, channels] = 0.0
#                     return output
#                 return hook
#             hook = create_freezer_hook(channels_freeze_list)
#             hook_handle = target_layer.register_forward_hook(hook)
#         with torch.no_grad():
#             outputs = self.model(inputs)
#             loss = criterion(outputs, labels)
#         if hook_handle is not None:
#             hook_handle.remove()
#         return loss.item()




import torch
from itertools import combinations

class channelstoolcdacp:

    def __init__(self, args, total_batches, device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.total_batches = total_batches
        self.base_rate = getattr(args, 'purning_base', 0.2)
        self.p_min = getattr(args, 'purning_min', 0.1)
        self.p_max = getattr(args, 'purning_max', 0.3)
        self.device = device
        
        # 用于存储历史分数的变量
        self.historical_channel_scores = None
        self.historical_batch_quality = None
        self.current_batch_idx = 0

    def _calculate_instantaneous_scores(self, feature_maps, labels):

        num_channels = feature_maps.shape[1]
        classes = torch.unique(labels)
        
        # 如果一个批次中只有一个类别，无法计算类间分离度，直接返回零分
        if len(classes) < 2:
            return torch.zeros(num_channels, device=self.device)

        intra_class_scores = torch.zeros(num_channels, device=self.device)
        inter_class_scores = torch.zeros(num_channels, device=self.device)

        # 1. 计算每个类别的质心 (Eq. 1)
        centroids = {}
        for c in classes:
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            class_feature_maps = feature_maps[class_indices]
            centroids[c.item()] = torch.mean(class_feature_maps, dim=0)

        # 2. 计算 Intra-class Compactness (Eq. 2)
        for c in classes:
            c_item = c.item()
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            class_feature_maps = feature_maps[class_indices]
            dist = class_feature_maps - centroids[c_item].unsqueeze(0)
            # 按通道计算 Frobenius 范数的平方
            compactness_per_sample = torch.sum(dist.pow(2), dim=[2, 3])
            avg_compactness_for_class = torch.mean(compactness_per_sample, dim=0)
            intra_class_scores += avg_compactness_for_class
        intra_class_scores /= len(classes) # 值越小越好

        # 3. 计算 Inter-class Separability (Eq. 3)
        class_ids = [c.item() for c in classes]
        for c1, c2 in combinations(class_ids, 2):
            dist = centroids[c1] - centroids[c2]
            # 按通道计算 Frobenius 范数 (注意论文中没有开方)
            separability = torch.sum(dist.pow(2), dim=[1, 2])
            inter_class_scores += torch.sqrt(separability) # Eq. 3 使用的是Frobenius范数，而不是其平方
        # 值越大越好

        # 4. 归一化和合并分数 (Eq. 4, 5, 6)
        norm_intra = intra_class_scores / (torch.sum(intra_class_scores))
        norm_inter = inter_class_scores / (torch.sum(inter_class_scores))
        
        instantaneous_scores = norm_inter - norm_intra
        return instantaneous_scores

    def _update_and_get_composite_scores(self, feature_maps, labels):
        """根据 Eq. (7), (8) 更新并计算复合重要性分数 I_i(t)"""
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps, labels)
        
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)

        # 计算历史平均分数
        historical_avg = self.historical_channel_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        
        # 计算动态权重 alpha (Eq. 8)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        
        # 计算复合分数 (Eq. 7)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        
        # 更新历史记录以备下次使用
        self.historical_channel_scores += instantaneous_scores
        
        return composite_scores

    def _calculate_pruning_rate(self, composite_scores):
        """根据 DACP 逻辑 (Eq. 9-12) 计算当前批次的剪枝率 P(t)"""
        batch_quality_score = torch.mean(composite_scores)
        
        # (Eq. 10)
        if self.current_batch_idx == 0:
            historical_avg_quality = batch_quality_score 
        else:
            historical_avg_quality = self.historical_batch_quality
        
            # 计算缩放因子 W(t) (Eq. 11)
    # 计算缩放因子 W(t) (Eq. 11)
        if historical_avg_quality > 0 and batch_quality_score < 0:
            # 历史表现好但当前表现差 -> 激进剪枝
            pruning_rate = self.p_max
        elif historical_avg_quality < 0 and batch_quality_score > 0:
            # 历史表现差但当前表现好 -> 保守剪枝
            pruning_rate = self.p_min
        else:
            # 其他情况（例如，两者同号）按照原始公式计算
            # 计算缩放因子 W(t) (Eq. 11)
            scaling_factor = historical_avg_quality / (batch_quality_score)
            pruning_rate = self.base_rate * scaling_factor.item()
            
        # 确保剪枝率不会是负数
        pruning_rate = max(0.0, pruning_rate)
        
        # 最终将剪枝率约束在 [p_min, p_max] 范围内
        pruning_rate = max(self.p_min, min(pruning_rate, self.p_max))

        # 更新历史批次质量的平均值
        if self.historical_batch_quality is None:
            self.historical_batch_quality = batch_quality_score
        else:
            self.historical_batch_quality = (self.historical_batch_quality * self.current_batch_idx + batch_quality_score) / (self.current_batch_idx + 1)
            
        return pruning_rate
        
    def _get_mask_for_batch(self, composite_scores, pruning_rate):
        """根据复合分数和剪枝率生成二进制掩码"""
        num_channels = len(composite_scores)
        num_to_prune = int(pruning_rate * num_channels)
        
        if num_to_prune == 0:
            return torch.ones(1, num_channels, 1, 1, device=self.device)
            
        # 对通道重要性分数进行排序，找到最不重要的通道
        ranking = torch.argsort(composite_scores)
        channels_to_prune = ranking[:num_to_prune]
        
        # 创建掩码
        mask = torch.ones(num_channels, device=self.device)
        mask[channels_to_prune] = 0
        
        # Reshape for broadcasting
        return mask.view(1, -1, 1, 1)

    def prune_channels(self, feature_maps, labels):
        """
        对给定的特征图执行CDACP剪枝。
        这是提供给客户端的主要接口。
        """
        # 确保数据在正确的设备上
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)

        # 1. 计算复合重要性分数
        composite_scores = self._update_and_get_composite_scores(feature_maps.detach(), labels)
        
        # 2. 计算当前批次的动态剪枝率
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        
        # 3. 获取剪枝掩码
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        
        # 4. 应用掩码
        # detach() 防止剪枝操作影响客户端模型的梯度计算
        pruned_feature_maps = feature_maps.detach() * mask
        
        # 更新批次计数器
        self.current_batch_idx += 1
        
        return pruned_feature_maps, pruning_rate
