import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from channelstools.channelstoolslcdacp import channelstoolcdacp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
class clientslcdacp(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.train_cnt=0
        self.pruning_tool_name = args.prune_tool
        self.fixed_alpha = args.fixed_alpha
        self.up_optimizer = None # 初始化 up_optimizer 屬性

    def split_train(self, up_model, cdacp):
        self.pruning_rates_history = []
        
        trainloader = self.load_train_data()
        
        self.model.train()
        up_model.train() 

        # [核心修改] 檢查優化器是否已創建，如果未創建（即第一次調用），則進行初始化
        if self.up_optimizer is None:
            if self.optimizer_str=="SGD":
                if self.pruning_tool_name=="default_mask_grad_momentum_0.7":
                    self.up_optimizer = torch.optim.SGD(up_model.parameters(), lr=self.learning_rate, momentum=0.7, weight_decay=5e-4)
                else:
                    self.up_optimizer = torch.optim.SGD(up_model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
            elif self.optimizer_str=="Adam":
                self.up_optimizer = torch.optim.AdamW(up_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        if self.pruning_tool_name =='default_mask_grad_scheduler':
            warmup_scheduler = LinearLR(
                self.up_optimizer, 
                start_factor=0.01, # 从 BASE_LR * 0.01 开始
                total_iters=5
            )
            main_scheduler = CosineAnnealingLR(
                self.up_optimizer, 
                T_max=200 - 5,
                eta_min=0  # 学习率最低会降到0
            )
            self.up_scheduler = SequentialLR(
                self.up_optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[5]
            )
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                down_output = self.model(x)
                # 1. 前向傳播剪枝
                if self.pruning_tool_name == 'historical':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_historical_only(down_output, y)
                elif self.pruning_tool_name == 'instantaneous':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_instantaneous_only(down_output, y)
                elif self.pruning_tool_name == 'fixed_alpha':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_fixed_alpha(down_output, y, self.fixed_alpha)
                elif self.pruning_tool_name == 'random':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_randomly(down_output, y)
                elif self.pruning_tool_name == 'top-k':
                    pruned_data, current_pruning_rate = cdacp.prune_top_k(down_output, y)
                elif self.pruning_tool_name == 'bath_top-k':
                    pruned_data, current_pruning_rate = cdacp.prune_by_batch_magnitude(down_output, y)
                elif self.pruning_tool_name == 'index_prune':
                    pruned_data, current_pruning_rate = cdacp.prune_by_channel_index(down_output, y)
                elif self.pruning_tool_name == 'default_division':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_division(down_output, y)        
                elif self.pruning_tool_name == 'default_recent_10':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_recent_10(down_output, y)   
                elif self.pruning_tool_name == 'default_recent_100':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_recent_100(down_output, y)   
                elif self.pruning_tool_name == 'STD':
                    pruned_data, current_pruning_rate = cdacp.prune_by_variance(down_output, y)  
                elif self.pruning_tool_name =='rand_top-k':
                    pruned_data, current_pruning_rate = cdacp.prune_by_probabilistic_magnitude(down_output, y)  
                elif self.pruning_tool_name =='default_SLIP':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_SLIP(down_output, y)  
                elif self.pruning_tool_name=='default_hybrid_history':   
                    pruned_data, current_pruning_rate = cdacp.prune_channels_hybrid_history(down_output, y)

                # --- 调用最终版的滑动窗口函数 ---
                elif self.pruning_tool_name == 'default_recent_10_rounds_mask_grad':
                    pruned_data, current_pruning_rate = cdacp.default_recent_10_rounds_mask_grad(down_output, y)

                elif self.pruning_tool_name == 'default_mask':
                    pruned_data, current_pruning_rate = cdacp.prune_channels(down_output, y)
                elif self.pruning_tool_name == 'default_keep_count_mask':
                    pruned_data, current_pruning_rate = cdacp.prune_channels(down_output, y)
                elif self.pruning_tool_name == 'default_mask_gradient':
                    pruned_data, current_pruning_rate = cdacp.prune_channels(down_output, y)
                elif self.pruning_tool_name == 'default_mask_keep_counts_grad':
                    pruned_data, current_pruning_rate = cdacp.prune_channels(down_output, y)
                elif self.pruning_tool_name == 'default_mask_grad_momentum_0.7':
                    pruned_data, current_pruning_rate = cdacp.prune_channels(down_output, y)   
                elif self.pruning_tool_name == 'STD_mask_grad':
                    pruned_data, current_pruning_rate = cdacp.prune_by_variance(down_output, y)        
                elif self.pruning_tool_name == 'rand_top-k_mask_grad':
                    pruned_data, current_pruning_rate = cdacp.prune_by_probabilistic_magnitude(down_output, y)          
                elif self.pruning_tool_name == 'fixed_alpha_mask_grad':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_fixed_alpha(down_output, y, self.fixed_alpha)        
                elif self.pruning_tool_name == 'uniform_quant':
                    pruned_data, current_pruning_rate = cdacp.prune_by_uniform_quantization(down_output, y)
                elif self.pruning_tool_name == 'default_bath_STD_mask_gradient':
                    pruned_data, current_pruning_rate = cdacp.prune_batch_STD(down_output, y)         
                elif self.pruning_tool_name == 'default_bath_value_mask_gradient':
                    pruned_data, current_pruning_rate = cdacp.prune_batch_value(down_output, y)                
                elif self.pruning_tool_name == 'default_bath_value_mask_keep_counts_gradient':
                    pruned_data, current_pruning_rate = cdacp.prune_batch_value(down_output, y)      
                elif self.pruning_tool_name == 'index_prune_mask_gradient':
                    pruned_data, current_pruning_rate = cdacp.prune_by_channel_index(down_output, y)        
                elif self.pruning_tool_name == 'index_rand_prune_mask_gradient':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_fixed_random(down_output, y)      
                elif self.pruning_tool_name == 'default_dot_mask_keep_counts_grad':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_dot(down_output, y)  
                elif self.pruning_tool_name == 'default_dot_mask_grad':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_dot(down_output, y)  
                elif self.pruning_tool_name == 'default_dot_mask_grad_min_max':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_dot_min_max(down_output, y)  
                elif self.pruning_tool_name == 'default_dot_mask_keep_countsgrad_min_max':
                    pruned_data, current_pruning_rate = cdacp.prune_channels_dot_min_max(down_output, y)  
                else: 
                    pruned_data, current_pruning_rate = cdacp.prune_channels(down_output, y)
                
                self.pruning_rates_history.append(current_pruning_rate)
                
                output = up_model(pruned_data)
                loss = self.loss(output, y)
                
                self.optimizer.zero_grad()
                self.up_optimizer.zero_grad() # 修改為 self.up_optimizer
                
                loss.backward()
                
                # [核心修正] 當策略為帶有 _grad 後綴的方法時，執行手動梯度掩碼
                if self.pruning_tool_name in ['default_mask_gradient', 'default_mask_keep_counts_grad', 'default_recent_10_rounds_mask_grad','default_recent_1_rounds_mask_grad','default_mask_grad_momentum_0.7','default_mask_grad_scheduler','STD_mask_grad','rand_top-k_mask_grad','fixed_alpha_mask_grad','default_bath_STD_mask_gradient','default_bath_value_mask_gradient','default_bath_value_mask_keep_counts_gradient','index_prune_mask_gradient','index_rand_prune_mask_gradient','default_dot_mask_keep_counts_grad','default_dot_mask_grad_min_max','default_dot_mask_keep_countsgrad_min_max']:
                    last_mask = cdacp.get_last_mask()
                    if last_mask is not None:
                        last_layer = self.model[-1] 
                        if isinstance(last_layer, (torch.nn.Conv2d, torch.nn.Linear)):
                            grad_mask = last_mask.squeeze().view(-1, 1, 1, 1) 
                            
                            if last_layer.weight.grad is not None:
                                last_layer.weight.grad.mul_(grad_mask)
                            
                            if last_layer.bias is not None and last_layer.bias.grad is not None:
                                last_layer.bias.grad.mul_(last_mask.squeeze())
                
                self.optimizer.step()
                self.up_optimizer.step() 
                if self.pruning_tool_name =='default_mask_grad_scheduler':
                    self.scheduler.step()
                    self.up_scheduler.step()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return up_model, self.model, self.pruning_rates_history
    
    def get_train_bath_num(self):
        trainloader = self.load_train_data()
        max_local_epochs = self.local_epochs
        num_batches_per_epoch = len(trainloader)
        return num_batches_per_epoch * max_local_epochs