import torch
from itertools import combinations
import random

class channelstoolcdacp:

    def __init__(self, args, total_batches, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.total_batches = total_batches
        self.base_rate = args.purning_base
        self.p_min = args.purning_min
        self.p_max = args.purning_max
        self.device = device
        self.historical_channel_scores = None
        self.historical_batch_quality = None
        self.current_batch_idx = 0
        self.channel_kept_counts = None
        self.historical_channel_scores_division = None
        self.recent_channel_scores = []
        self.recent_batch_qualities = []
        self.recent_batch_qualities_for_rate = []
        self.recent_channel_scores_100 = []
        self.last_generated_mask = None
    
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
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
        historical_avg = self.historical_channel_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_channel_scores += instantaneous_scores
        return composite_scores
        
    def _update_and_get_composite_scores_division(self, feature_maps, labels):
        instantaneous_scores = self._calculate_instantaneous_scores_division(feature_maps, labels)
        if self.historical_channel_scores_division is None:
            self.historical_channel_scores_division = torch.zeros_like(instantaneous_scores)
        historical_avg = self.historical_channel_scores_division / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        alpha = 1.0 - (self.current_batch_idx / self.total_batches)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_channel_scores_division += instantaneous_scores
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
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
        self.historical_channel_scores += instantaneous_scores
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
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
        self.historical_channel_scores += instantaneous_scores
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

    def default_mask_no_gradient(self, feature_maps, labels):
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
        if not self.recent_batch_qualities:
            recent_avg_quality = current_batch_quality
        else:
            recent_qualities_tensor = torch.stack(self.recent_batch_qualities)
            recent_avg_quality = torch.mean(recent_qualities_tensor)
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
        if not self.recent_batch_qualities_for_rate:
            recent_avg_quality = current_batch_quality
        else:
            recent_qualities_tensor = torch.stack(self.recent_batch_qualities_for_rate)
            recent_avg_quality = torch.mean(recent_qualities_tensor)
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
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
        self.historical_channel_scores += instantaneous_scores
        historical_avg_scores = self.historical_channel_scores / (self.current_batch_idx + 1)
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
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
        self.historical_channel_scores += instantaneous_scores
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
        if self.historical_channel_scores is None:
            self.historical_channel_scores = torch.zeros_like(instantaneous_scores)
            historical_avg = torch.zeros_like(instantaneous_scores)
        else:
            historical_avg = self.historical_channel_scores / self.current_batch_idx if self.current_batch_idx > 0 else torch.zeros_like(instantaneous_scores)
        composite_scores = alpha * instantaneous_scores + (1 - alpha) * historical_avg
        self.historical_channel_scores += instantaneous_scores
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