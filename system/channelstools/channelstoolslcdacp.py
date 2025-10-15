import torch
from itertools import combinations
import random

class UniformQuantizationWithPruningRate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, pruning_rate=0.1):
        compression_ratio = 1.0 - pruning_rate
        original_bits = 32
        quantized_bits = max(1, round(original_bits * compression_ratio))
        ctx.quantized_bits = quantized_bits
        min_val = input_tensor.min()
        max_val = input_tensor.max()
        if max_val == min_val:
            return input_tensor
        num_levels = 2**quantized_bits
        scale = (max_val - min_val) / (num_levels - 1)
        quantized_tensor = torch.round((input_tensor - min_val) / scale)
        dequantized_tensor = quantized_tensor * scale + min_val
        return dequantized_tensor

    @staticmethod
    def backward(ctx, grad_output):
        quantized_bits = ctx.quantized_bits
        if grad_output is None:
            return None, None
        min_val = grad_output.min()
        max_val = grad_output.max()
        if max_val == min_val:
            return grad_output, None
        num_levels = 2**quantized_bits
        scale = (max_val - min_val) / (num_levels - 1)
        quantized_grad = torch.round((grad_output - min_val) / scale)
        dequantized_grad = quantized_grad * scale + min_val
        return dequantized_grad, None

uniform_quantize_by_pruning = UniformQuantizationWithPruningRate.apply


class channelstoolcdacp:

    def __init__(self, args, total_batches, one_round_batches, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.total_batches = total_batches
        self.one_round_batches = one_round_batches
        self.base_rate = args.purning_base
        self.p_min = args.purning_min
        self.p_max = args.purning_max
        self.device = device
        
        self.historical_scores = None
        
        self.historical_batch_quality = None
        self.current_batch_idx = 0
        self.channel_kept_counts = None
        self.recent_channel_scores = []
        self.recent_batch_qualities = []
        self.recent_batch_qualities_for_rate = []
        self.recent_channel_scores_100 = []
        self.last_generated_mask = None
        self.fixed_pruning_indices = None
        
        self._batch_history_for_1_round = []
        self._batch_history_for_2_round = []
        self._batch_history_for_10_round = []

    def get_last_mask(self):
        return self.last_generated_mask

    def default_mask(self, pruning_percentage=None):
        if self.channel_kept_counts is None:
            print("警告: 尚未進行任何剪枝，無法生成最終掩碼。將返回一個全1的掩碼。")
            return None 
        if pruning_percentage is None:
            pruning_percentage = self.base_rate
        keep_percentage = 1.0 - pruning_percentage
        num_channels = len(self.channel_kept_counts)
        num_to_keep = int(num_channels * keep_percentage)
        print(f"生成最終測試掩碼: 剪枝率={pruning_percentage:.2f}, 保留率={keep_percentage:.2f}")
        print(f"總通道數={num_channels}, 計劃保留={num_to_keep}")
        if num_to_keep >= num_channels:
            return torch.ones(1, num_channels, 1, 1, device=self.device)
        if num_to_keep <= 0:
            return torch.zeros(1, num_channels, 1, 1, device=self.device)
        _, top_indices = torch.topk(self.channel_kept_counts, k=num_to_keep)
        final_mask = torch.zeros(num_channels, device=self.device)
        final_mask[top_indices] = 1.0
        return final_mask.view(1, -1, 1, 1)

    def _update_pruning_stats(self, mask):
        if mask is None: return
        num_channels = mask.shape[1]
        if self.channel_kept_counts is None:
            self.channel_kept_counts = torch.zeros(num_channels, device=self.device)
        kept_indicators = mask.squeeze().float()
        if kept_indicators.dim() == 1:
            update_values = kept_indicators
        else:
            update_values = torch.sum(kept_indicators, dim=0)
        self.channel_kept_counts += update_values.detach()

    def get_kept_channel_counts(self):
        if self.channel_kept_counts is None:
            print("尚未進行任何剪枝。")
            return {}
        kept_counts_dict = {i: int(count.item()) for i, count in enumerate(self.channel_kept_counts)}
        return kept_counts_dict

    def _calculate_instantaneous_scores_division(self, feature_maps, labels):
        num_channels = feature_maps.shape[1]
        classes = torch.unique(labels)
        if len(classes) < 2:
            return torch.ones(num_channels, device=self.device)
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
        sum_intra = torch.sum(intra_class_scores)
        sum_inter = torch.sum(inter_class_scores)
        if sum_intra == 0 or sum_inter == 0:
            return torch.ones(num_channels, device=self.device)
        norm_intra = intra_class_scores / sum_intra
        norm_inter = inter_class_scores / sum_inter
        instantaneous_scores = norm_inter / (norm_intra)
        return instantaneous_scores

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
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_scores += instantaneous_scores
        return composite_scores
        
    def _update_and_get_composite_scores_division(self, feature_maps, labels):
        instantaneous_scores = self._calculate_instantaneous_scores_division(feature_maps, labels)
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_scores += instantaneous_scores
        return composite_scores
        
    def _update_and_get_composite_scores_recent_10(self, feature_maps, labels):
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps, labels)
        self.recent_channel_scores.append(instantaneous_scores)
        if len(self.recent_channel_scores) > 10:
            self.recent_channel_scores.pop(0)
        if not self.recent_channel_scores:
            historical_avg = torch.zeros_like(instantaneous_scores)
        else:
            historical_avg = torch.mean(torch.stack(self.recent_channel_scores), dim=0)
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        self.historical_scores += instantaneous_scores
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        return composite_scores

    def _update_and_get_composite_scores_recent_100(self, feature_maps, labels):
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps, labels)
        self.recent_channel_scores_100.append(instantaneous_scores)
        if len(self.recent_channel_scores_100) > 100:
            self.recent_channel_scores_100.pop(0)
        if not self.recent_channel_scores_100:
            historical_avg = torch.zeros_like(instantaneous_scores)
        else:
            historical_avg = torch.mean(torch.stack(self.recent_channel_scores_100), dim=0)
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        self.historical_scores += instantaneous_scores
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
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
            scaling_factor = historical_avg_quality / (batch_quality_score ) 
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

    def default_recent_10_rounds_mask_grad(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        window_size_in_batches = self.one_round_batches * 10
        batch_history = self._batch_history_for_10_round
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        batch_history.append(instantaneous_scores)
        if window_size_in_batches > 0 and len(batch_history) > window_size_in_batches:
            batch_history.pop(0)
        historical_avg = torch.mean(torch.stack(batch_history), dim=0) if batch_history else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
        
    def default_recent_1_rounds_mask_grad(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        window_size_in_batches = self.one_round_batches * 1
        batch_history = self._batch_history_for_1_round
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        batch_history.append(instantaneous_scores)
        if window_size_in_batches > 0 and len(batch_history) > window_size_in_batches:
            batch_history.pop(0)
        historical_avg = torch.mean(torch.stack(batch_history), dim=0) if batch_history else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate

    def prune_channels(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores(feature_maps.detach(), labels)
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate

    def prune_channels_SLIP(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        current_batch_quality = torch.mean(instantaneous_scores)
        self.recent_batch_qualities.append(current_batch_quality.detach())
        if len(self.recent_batch_qualities) > 100:
            self.recent_batch_qualities.pop(0)
        recent_avg_quality = torch.mean(torch.stack(self.recent_batch_qualities)) if self.recent_batch_qualities else current_batch_quality
        p_schedule = self.p_min + (self.p_max - self.p_min) * (self.current_batch_idx / self.total_batches)
        if recent_avg_quality > 0 and current_batch_quality < 0:
            pruning_rate = self.p_max
        elif recent_avg_quality < 0 and current_batch_quality > 0:
            pruning_rate = self.p_min
        else:
            scaling_factor = recent_avg_quality / current_batch_quality
            pruning_rate = p_schedule * scaling_factor.item()
        pruning_rate = max(self.p_min, min(pruning_rate, self.p_max))
        mask = self._get_mask_for_batch(instantaneous_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate

    def prune_channels_hybrid_history(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores(feature_maps.detach(), labels)
        current_batch_quality = torch.mean(composite_scores)
        self.recent_batch_qualities_for_rate.append(current_batch_quality.detach())
        if len(self.recent_batch_qualities_for_rate) > 100:
            self.recent_batch_qualities_for_rate.pop(0)
        recent_avg_quality = torch.mean(torch.stack(self.recent_batch_qualities_for_rate)) if self.recent_batch_qualities_for_rate else current_batch_quality
        p_schedule = self.p_min + (self.p_max - self.p_min) * (self.current_batch_idx / self.total_batches)
        if recent_avg_quality > 0 and current_batch_quality < 0:
            pruning_rate = self.p_max
        elif recent_avg_quality < 0 and current_batch_quality > 0:
            pruning_rate = self.p_min
        else:
            scaling_factor = recent_avg_quality / current_batch_quality
            pruning_rate = p_schedule * scaling_factor
        pruning_rate = max(self.p_min, min(pruning_rate, self.p_max))
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate

    def prune_channels_division(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores_division(feature_maps.detach(), labels)
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
        
    def prune_channels_recent_10(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores_recent_10(feature_maps.detach(), labels)
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate

    def prune_channels_recent_100(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores_recent_100(feature_maps.detach(), labels)
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate

    def prune_channels_historical_only(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        self.historical_scores += instantaneous_scores
        historical_avg_scores = self.historical_scores / (self.current_batch_idx + 1)
        pruning_rate = self._calculate_pruning_rate(historical_avg_scores)
        mask = self._get_mask_for_batch(historical_avg_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
    
    def prune_channels_instantaneous_only(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        self.historical_scores += instantaneous_scores
        pruning_rate = self._calculate_pruning_rate(instantaneous_scores)
        mask = self._get_mask_for_batch(instantaneous_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
    
    def prune_channels_fixed_alpha(self, feature_maps, labels, alpha):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        instantaneous_scores = self._calculate_instantaneous_scores(feature_maps.detach(), labels)
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_scores += instantaneous_scores
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
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
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.last_generated_mask = mask
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
        self._update_pruning_stats(mask_4d)
        pruned_feature_maps = feature_maps * mask_4d
        self.last_generated_mask = mask_4d
        return pruned_feature_maps, pruning_rate
    
    def prune_by_batch_magnitude(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        N, C, H, W = feature_maps.shape
        pruning_rate = self.base_rate
        num_to_prune = int(pruning_rate * C)
        if num_to_prune <= 0:
            mask = torch.ones(1, C, 1, 1, device=self.device)
        else:
            batch_channel_scores = torch.sum(feature_maps.detach().pow(2), dim=[0, 2, 3])
            ranking = torch.argsort(batch_channel_scores)
            channels_to_prune = ranking[:num_to_prune]
            mask_1d = torch.ones(C, device=self.device)
            mask_1d[channels_to_prune] = 0
            mask = mask_1d.view(1, C, 1, 1)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
    
    def prune_by_channel_index(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        N, C, H, W = feature_maps.shape
        pruning_rate = self.base_rate
        num_to_prune = int(pruning_rate * C)
        if num_to_prune <= 0:
            mask = torch.ones(1, C, 1, 1, device=self.device)
            pruned_feature_maps = feature_maps
        else:
            channels_to_prune = torch.arange(num_to_prune, device=self.device)
            mask_1d = torch.ones(C, device=self.device)
            mask_1d[channels_to_prune] = 0
            mask = mask_1d.view(1, C, 1, 1)
            pruned_feature_maps = feature_maps * mask
        self._update_pruning_stats(mask)
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
    
    def prune_by_variance(self, feature_maps, labels=None):
        feature_maps = feature_maps.to(self.device)
        N, C, H, W = feature_maps.shape
        pruning_rate = self.base_rate
        percentage_to_keep = 1.0 - pruning_rate
        num_to_keep = int(C * percentage_to_keep)
        if num_to_keep >= C:
            mask_4d = torch.ones(N, C, 1, 1, device=self.device)
            self.last_generated_mask = mask_4d
            return feature_maps, 0.0
        if num_to_keep <= 0:
            mask_4d = torch.zeros(N, C, 1, 1, device=self.device)
            self._update_pruning_stats(mask_4d)
            self.last_generated_mask = mask_4d
            return feature_maps * mask_4d, 1.0
        channel_variances = torch.var(feature_maps.detach(), dim=(-2, -1), unbiased=False)
        top_indices = torch.topk(channel_variances, k=num_to_keep, dim=1).indices
        mask_2d = torch.zeros_like(channel_variances, device=self.device)
        mask_2d.scatter_(dim=1, index=top_indices, value=1.0)
        mask_4d = mask_2d.view(N, C, 1, 1)
        self._update_pruning_stats(mask_4d)
        pruned_feature_maps = feature_maps * mask_4d
        self.last_generated_mask = mask_4d
        return pruned_feature_maps, pruning_rate

    def prune_by_probabilistic_magnitude(self, feature_maps, labels=None):
        feature_maps = feature_maps.to(self.device)
        N, C, H, W = feature_maps.shape
        pruning_rate = self.base_rate
        num_to_prune = int(C * pruning_rate)
        if num_to_prune >= C:
            mask_4d = torch.zeros(N, C, 1, 1, device=self.device)
            self.last_generated_mask = mask_4d
            return mask_4d, 1.0
        if num_to_prune <= 0:
            mask_4d = torch.ones(N, C, 1, 1, device=self.device)
            self.last_generated_mask = mask_4d
            return feature_maps, 0.0
        magnitudes = torch.sum(feature_maps.detach().pow(2), dim=(-2, -1))
        pruning_scores = -magnitudes + 1e-9
        pruning_probs = torch.nn.functional.softmax(pruning_scores, dim=1)
        channels_to_prune = torch.multinomial(pruning_probs, num_samples=num_to_prune, replacement=False)
        mask_2d = torch.ones_like(magnitudes, device=self.device)
        mask_2d.scatter_(dim=1, index=channels_to_prune, value=0.0)
        mask_4d = mask_2d.view(N, C, 1, 1)
        self._update_pruning_stats(mask_4d)
        pruned_feature_maps = feature_maps * mask_4d
        self.last_generated_mask = mask_4d
        return pruned_feature_maps, pruning_rate
    
    def _update_and_get_composite_scores_std(self, feature_maps):
        instantaneous_scores = torch.std(feature_maps.detach(), dim=[0, 2, 3])
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_scores += instantaneous_scores
        return composite_scores
    
    def prune_batch_STD(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores_std(feature_maps)
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
    
    def prune_by_uniform_quantization(self, feature_maps, labels=None):
        pruning_rate = self.base_rate 
        if pruning_rate == 0.0:
            return feature_maps, 0.0
        pruned_feature_maps = uniform_quantize_by_pruning(feature_maps, pruning_rate)
        self.last_generated_mask = None 
        return pruned_feature_maps, pruning_rate
        
    def _update_and_get_composite_scores_value(self, feature_maps):
        instantaneous_scores = torch.sum(feature_maps.detach().pow(2), dim=[0, 2, 3])
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)
        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_scores += instantaneous_scores
        return composite_scores

    def prune_batch_value(self, feature_maps, labels):
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        composite_scores = self._update_and_get_composite_scores_value(feature_maps)
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
        
    def prune_channels_fixed_random(self, feature_maps, labels=None):
        feature_maps = feature_maps.to(self.device)
        N, C, H, W = feature_maps.shape
        pruning_rate = self.base_rate
        if self.fixed_pruning_indices is None:
            print("--- [首次运行] 固定随机剪枝策略初始化 ---")
            num_to_prune = int(pruning_rate * C)
            if num_to_prune > 0:
                all_channel_indices = list(range(C))
                self.fixed_pruning_indices = random.sample(all_channel_indices, num_to_prune)
                print(f"总通道数: {C}, 剪枝率: {pruning_rate:.2f}, 计划剪枝: {num_to_prune} 个通道。")
            else:
                print("剪枝数量为0，不进行任何剪枝。")
                self.fixed_pruning_indices = []
        if self.fixed_pruning_indices and len(self.fixed_pruning_indices) > 0:
            mask_1d = torch.ones(C, device=self.device)
            mask_1d[self.fixed_pruning_indices] = 0
            mask = mask_1d.view(1, C, 1, 1)
        else:
            mask = torch.ones(1, C, 1, 1, device=self.device)
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate

    # ================================================================= #
    # ==================== 新增剪枝方法 (纯添加) ======================== #
    # ================================================================= #
    
    def _calculate_instantaneous_scores_dot(self, feature_maps, labels, temperature=0.1):
        """
        [新增] 基于点积相似度计算瞬时分数，并使用Softmax进行非线性归一化。
        此版本为纯新增，不更改任何原有代码。
        """
        num_channels = feature_maps.shape[1]
        classes = torch.unique(labels)

        if len(classes) < 2:
            return torch.zeros(num_channels, device=self.device)

        # 将特征图展平以便进行点积运算
        feature_maps_vec = feature_maps.view(feature_maps.shape[0], num_channels, -1)

        # --- 计算类中心 ---
        centroids = {}
        for c in classes:
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            centroids[c.item()] = torch.mean(feature_maps_vec[class_indices], dim=0)

        # --- 重新定义: 类内紧凑度 (Intra-Class Compactness) ---
        intra_class_scores = torch.zeros(num_channels, device=self.device)
        for c in classes:
            c_item = c.item()
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            class_feature_maps_vec = feature_maps_vec[class_indices]
            centroid_vec = centroids[c_item].unsqueeze(0)
            dot_products = torch.mean(torch.sum(class_feature_maps_vec * centroid_vec, dim=2), dim=0)
            intra_class_scores += dot_products
        
        intra_class_scores /= len(classes)

        # --- 重新定义: 类间分离度 (Inter-Class Separability) ---
        inter_class_scores = torch.zeros(num_channels, device=self.device)
        class_ids = [c.item() for c in classes]
        
        if len(class_ids) > 1:
            num_pairs = 0
            for c1, c2 in combinations(class_ids, 2):
                centroid1_vec = centroids[c1]
                centroid2_vec = centroids[c2]
                dot_product_centroids = torch.sum(centroid1_vec * centroid2_vec, dim=1)
                inter_class_scores += dot_product_centroids
                num_pairs += 1
            inter_class_scores /= num_pairs

        # --- 最终瞬时分数 ---
        instantaneous_scores = intra_class_scores - inter_class_scores
        
        # --- 使用带温度的Softmax进行归一化 ---
        if temperature > 0:
            normalized_scores = torch.nn.functional.softmax(instantaneous_scores / temperature, dim=0)
        else: # 避免除以零
            normalized_scores = torch.nn.functional.softmax(instantaneous_scores, dim=0)
            
        return normalized_scores

    def _update_and_get_composite_scores_dot(self, feature_maps, labels, temperature):
        """
        [新增] 调用新的点积评分标准，并结合历史分数。
        """
        instantaneous_scores = self._calculate_instantaneous_scores_dot(feature_maps, labels, temperature)
        
        # 复用已有的 self.historical_scores 属性
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)

        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        
        self.historical_scores += instantaneous_scores
        return composite_scores

    def prune_channels_dot(self, feature_maps, labels, temperature=0.1):
        """
        [新增] 最终对外调用的新剪枝方法，名为 prune_channels_dot。
        完全不影响任何原有代码。
        """
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        
        # 调用全新的、独立的评分和组合函数
        composite_scores = self._update_and_get_composite_scores_dot(feature_maps.detach(), labels, temperature)
        
        # 复用已有的剪枝率计算和掩码生成逻辑
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate



        # ================================================================= #
    # ============ 新增剪枝方法 (Min-Max Dot Product) =================== #
    # ================================================================= #
    
    def _calculate_instantaneous_scores_dot_min_max(self, feature_maps, labels):
        """
        [新增] 基于点积相似度计算瞬时分数，并使用 Min-Max 进行线性归一化。
        此版本为 prune_channels_dot 的变体。
        """
        num_channels = feature_maps.shape[1]
        classes = torch.unique(labels)

        if len(classes) < 2:
            return torch.zeros(num_channels, device=self.device)

        # 将特征图展平以便进行点积运算
        feature_maps_vec = feature_maps.view(feature_maps.shape[0], num_channels, -1)

        # --- 计算类中心 ---
        centroids = {}
        for c in classes:
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            centroids[c.item()] = torch.mean(feature_maps_vec[class_indices], dim=0)

        # --- 计算类内紧凑度 (Intra-Class Compactness) ---
        intra_class_scores = torch.zeros(num_channels, device=self.device)
        for c in classes:
            c_item = c.item()
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            class_feature_maps_vec = feature_maps_vec[class_indices]
            centroid_vec = centroids[c_item].unsqueeze(0)
            dot_products = torch.mean(torch.sum(class_feature_maps_vec * centroid_vec, dim=2), dim=0)
            intra_class_scores += dot_products
        
        intra_class_scores /= len(classes)

        # --- 计算类间分离度 (Inter-Class Separability) ---
        inter_class_scores = torch.zeros(num_channels, device=self.device)
        class_ids = [c.item() for c in classes]
        
        if len(class_ids) > 1:
            num_pairs = 0
            for c1, c2 in combinations(class_ids, 2):
                centroid1_vec = centroids[c1]
                centroid2_vec = centroids[c2]
                dot_product_centroids = torch.sum(centroid1_vec * centroid2_vec, dim=1)
                inter_class_scores += dot_product_centroids
                num_pairs += 1
            inter_class_scores /= num_pairs

        # --- 最终瞬时分数 ---
        instantaneous_scores = intra_class_scores - inter_class_scores
        
        # --- 使用 Min-Max 进行归一化 ---
        min_val = torch.min(instantaneous_scores)
        max_val = torch.max(instantaneous_scores)
        
        # 避免除以零的边界情况
        if max_val == min_val:
            return torch.zeros_like(instantaneous_scores)
            
        normalized_scores = (instantaneous_scores - min_val) / (max_val - min_val)
            
        return normalized_scores

    def _update_and_get_composite_scores_dot_min_max(self, feature_maps, labels):
        """
        [新增] 调用新的点积+Min-Max评分标准，并结合历史分数。
        """
        instantaneous_scores = self._calculate_instantaneous_scores_dot_min_max(feature_maps, labels)
        
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)

        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        
        self.historical_scores += instantaneous_scores
        return composite_scores

    def prune_channels_dot_min_max(self, feature_maps, labels):
        """
        [新增] 最终对外调用的新剪枝方法，名为 prune_channels_dot_min_max。
        """
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        
        # 调用全新的、独立的评分和组合函数
        composite_scores = self._update_and_get_composite_scores_dot_min_max(feature_maps.detach(), labels)
        
        # 复用已有的剪枝率计算和掩码生成逻辑
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate


    def _update_and_get_composite_scores_dot_fixed_alpha(self, feature_maps, labels, temperature, alpha):
        """
        [新增] 固定 alpha 值的 .dot 变体，用于结合历史分数。
        """
        instantaneous_scores = self._calculate_instantaneous_scores_dot(feature_maps, labels, temperature)
        
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)

        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        
        # 使用传入的固定 alpha 值，而不是动态计算
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        
        self.historical_scores += instantaneous_scores
        return composite_scores

    def prune_channels_dot_fixed_alpha(self, feature_maps, labels, temperature=0.1, alpha=0.5):
        """
        [新增] 最终对外调用的新剪枝方法，使用固定的 alpha 值和 softmax 归一化。
        """
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        
        # 调用使用固定 alpha 的新组合函数
        composite_scores = self._update_and_get_composite_scores_dot_fixed_alpha(feature_maps.detach(), labels, temperature, alpha)
        
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate
        
    def _update_and_get_composite_scores_dot_fixed_alpha_min_max(self, feature_maps, labels, alpha):
        """
        [新增] 固定 alpha 值的 .dot_min_max 变体，用于结合历史分数。
        """
        instantaneous_scores = self._calculate_instantaneous_scores_dot_min_max(feature_maps, labels)
        
        if self.historical_scores is None:
            self.historical_scores = torch.zeros_like(instantaneous_scores)

        historical_avg = self.historical_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        
        # 使用传入的固定 alpha 值
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        
        self.historical_scores += instantaneous_scores
        return composite_scores

    def prune_channels_dot_fixed_alpha_min_max(self, feature_maps, labels, alpha=0.5):
        """
        [新增] 最终对外调用的新剪枝方法，使用固定的 alpha 值和 min-max 归一化。
        """
        feature_maps = feature_maps.to(self.device)
        labels = labels.to(self.device)
        
        # 调用使用固定 alpha 的新组合函数
        composite_scores = self._update_and_get_composite_scores_dot_fixed_alpha_min_max(feature_maps.detach(), labels, alpha)
        
        pruning_rate = self._calculate_pruning_rate(composite_scores)
        mask = self._get_mask_for_batch(composite_scores, pruning_rate)
        
        self._update_pruning_stats(mask)
        pruned_feature_maps = feature_maps * mask
        
        self.current_batch_idx += 1
        self.last_generated_mask = mask
        return pruned_feature_maps, pruning_rate