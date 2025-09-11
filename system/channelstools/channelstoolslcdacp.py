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
import random

class channelstoolcdacp:

    def __init__(self, args, total_batches, device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.total_batches = total_batches
        self.base_rate =args.purning_base
        self.p_min = args.purning_min
        self.p_max = args.purning_max
        self.device = device

        # 用于DACP历史分数的变量
        self.historical_channel_scores = None
        self.historical_batch_quality = None
        self.current_batch_idx = 0

        self.channel_kept_counts = None
    
    def _update_pruning_stats(self, mask):
        """
        私有辅助函数，更新通道“被保留”的统计信息。
        """
        num_channels = mask.shape[1]
        if self.channel_kept_counts is None:
            self.channel_kept_counts = torch.zeros(num_channels, device=self.device)

        # 直接用 mask 来统计被保留的通道 (值为1的位置)
        kept_indicators = mask.squeeze().float()

        if kept_indicators.dim() == 1:
            # 批次级mask
            update_values = kept_indicators
        else:
            # 样本级mask, 沿批次维度求和
            update_values = torch.sum(kept_indicators, dim=0)
        
        self.channel_kept_counts += update_values.detach()

    def get_kept_channel_counts(self):

        if self.channel_kept_counts is None:
            print("尚未进行任何剪枝。")
            return {}

        kept_counts_dict = {i: int(count.item()) for i, count in enumerate(self.channel_kept_counts)}
        return kept_counts_dict



    def _calculate_instantaneous_scores(self, feature_maps, labels):
        num_channels = feature_maps.shape[1]
        classes = torch.unique(labels)
        
        if len(classes) < 2:
            return torch.zeros(num_channels, device=self.device)

        intra_class_scores = torch.zeros(num_channels, device=self.device)
        inter_class_scores = torch.zeros(num_channels, device=self.device)

        centroids = {}
        for c in classes:
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            class_feature_maps = feature_maps[class_indices]
            centroids[c.item()] = torch.mean(class_feature_maps, dim=0)

        for c in classes:
            c_item = c.item()
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            class_feature_maps = feature_maps[class_indices]
            dist = class_feature_maps - centroids[c_item].unsqueeze(0)
            compactness_per_sample = torch.sum(dist.pow(2), dim=[2, 3])
            avg_compactness_for_class = torch.mean(compactness_per_sample, dim=0)
            intra_class_scores += avg_compactness_for_class
        intra_class_scores /= len(classes)

        class_ids = [c.item() for c in classes]
        for c1, c2 in combinations(class_ids, 2):
            dist = centroids[c1] - centroids[c2]
            separability = torch.sum(dist.pow(2), dim=[1, 2])
            inter_class_scores += separability 

        # 防止分母为0
        sum_intra = torch.sum(intra_class_scores)
        sum_inter = torch.sum(inter_class_scores)
        if sum_intra == 0 or sum_inter == 0:
            return torch.zeros(num_channels, device=self.device)

        norm_intra = intra_class_scores / sum_intra
        norm_inter = inter_class_scores / sum_inter
        
        instantaneous_scores = norm_inter - norm_intra
        return instantaneous_scores

    def _update_and_get_composite_scores(self, feature_maps, labels):
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps, labels)
        
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)

        historical_avg = self.historical_channel_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        
        self.historical_channel_scores += instantaneous_scores
        
        return composite_scores

    def _calculate_pruning_rate(self, composite_scores):
        batch_quality_score = torch.mean(composite_scores)
        
        if self.current_batch_idx == 0:
            historical_avg_quality = batch_quality_score 
        else:
            historical_avg_quality = self.historical_batch_quality
        
        if historical_avg_quality > 0 and batch_quality_score < 0:
            pruning_rate = self.p_max
        elif historical_avg_quality < 0 and batch_quality_score > 0:
            pruning_rate = self.p_min
        else:
            if abs(batch_quality_score.item()) < 1e-8: # 避免除以0
                scaling_factor = 1.0
            else:
                scaling_factor = historical_avg_quality / batch_quality_score
            pruning_rate = self.base_rate * scaling_factor.item()
            
        pruning_rate = max(0.0, pruning_rate)
        
        pruning_rate = max(self.p_min, min(pruning_rate, self.p_max))

        if self.historical_batch_quality is None:
            self.historical_batch_quality = batch_quality_score
        else:
            self.historical_batch_quality = (self.historical_batch_quality * self.current_batch_idx + batch_quality_score) / (self.current_batch_idx + 1)
            
        return pruning_rate
        
    def _get_mask_for_batch(self, composite_scores, pruning_rate):
        num_channels = len(composite_scores)
        num_to_prune = int(pruning_rate * num_channels)
        
        if num_to_prune <= 0:
            return torch.ones(1, num_channels, 1, 1, device=self.device)
            
        ranking = torch.argsort(composite_scores)
        channels_to_prune = ranking[:num_to_prune]
        
        mask = torch.ones(num_channels, device=self.device)
        mask[channels_to_prune] = 0
        
        return mask.view(1, -1, 1, 1)


    def prune_channels(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores(feature_maps.detach(), labels)
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        
        self._update_pruning_stats(mask) # 更新统计
        
        pruned_feature_maps = feature_maps.detach() * mask
        self.current_batch_idx += 1
        return pruned_feature_maps, pruning_rate

    def prune_channels_historical_only(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
        self.historical_channel_scores += instantaneous_scores
        historical_avg_scores = self.historical_channel_scores / (self.current_batch_idx + 1)
        pruning_rate = self._calculate_pruning_rate(historical_avg_scores)
        mask = self._get_mask_for_batch(historical_avg_scores, pruning_rate)

        self._update_pruning_stats(mask) # 更新统计

        pruned_feature_maps = feature_maps.detach() * mask
        self.current_batch_idx += 1
        return pruned_feature_maps, pruning_rate
    
    def prune_channels_instantaneous_only(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
        self.historical_channel_scores += instantaneous_scores
        pruning_rate = self._calculate_pruning_rate(instantaneous_scores)
        mask = self._get_mask_for_batch(instantaneous_scores, pruning_rate)
        
        self._update_pruning_stats(mask) # 更新统计

        pruned_feature_maps = feature_maps.detach() * mask
        self.current_batch_idx += 1
        return pruned_feature_maps, pruning_rate
    
    def prune_channels_fixed_alpha(self, feature_maps, labels, alpha):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
            historical_avg = torch.zeros_like(instantaneous_scores)
        else:
            historical_avg = self.historical_channel_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_channel_scores += instantaneous_scores
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        
        self._update_pruning_stats(mask) # 更新统计

        pruned_feature_maps = feature_maps.detach() * mask
        self.current_batch_idx += 1
        return pruned_feature_maps, pruning_rate


    def prune_channels_randomly(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        num_channels = feature_maps.shape[1]
        pruning_rate = random.uniform(self.p_min, self.p_max)
        num_to_prune = int(pruning_rate * num_channels)
        if num_to_prune <= 0:
            mask = torch.ones(1, num_channels, 1, 1, device=self.device)
        else:
            all_channels = list(range(num_channels))
            channels_to_prune = random.sample(all_channels, num_to_prune)
            mask = torch.ones(num_channels, device=self.device)
            mask[channels_to_prune] = 0
            mask = mask.view(1, -1, 1, 1)
        
        self._update_pruning_stats(mask) # 更新统计
        
        pruned_feature_maps = feature_maps.detach() * mask
        return pruned_feature_maps, pruning_rate
        
    def prune_top_k(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        N, C, H, W = feature_maps.shape
        pruning_rate = self.base_rate
        num_to_prune = int(pruning_rate * C)
        if num_to_prune <= 0:
            mask_4d = torch.ones(N, C, 1, 1, device=self.device)
        else:
            channel_scores_per_sample = torch.sum(feature_maps.detach().pow(2), dim=[2, 3])
            ranking = torch.argsort(channel_scores_per_sample, dim=1)
            channels_to_prune = ranking[:, :num_to_prune]
            mask_2d = torch.ones_like(channel_scores_per_sample)
            mask_2d.scatter_(dim=1, index=channels_to_prune, value=0.0)
            mask_4d = mask_2d.view(N, C, 1, 1)

        self._update_pruning_stats(mask_4d) # 更新统计

        pruned_feature_maps = feature_maps * mask_4d
        return pruned_feature_maps, pruning_rate