import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from channelstools.channelstoolslcdacp import channelstoolcdacp
# from channelstools.channelstoolscdacp import channel_infomation

class clientslcdacp(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.train_cnt=0
        self.pruning_tool_name = args.prune_tool
        self.fixed_alpha = args.fixed_alpha



    def split_train(self, up_model, cdacp):
        self.pruning_rates_history = []
        
        trainloader = self.load_train_data()
        
        self.model.train()
        up_model.train() 
        up_optimizer = torch.optim.SGD(up_model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        
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
# 根据 self.pruning_tool_name 调用不同的剪枝方法
            if self.pruning_tool_name == 'historical':
                pruned_data, current_pruning_rate = cdacp.prune_channels_historical_only(down_output, y)
            elif self.pruning_tool_name == 'instantaneous':
                pruned_data, current_pruning_rate = cdacp.prune_channels_instantaneous_only(down_output, y)
            elif self.pruning_tool_name == 'fixed_alpha':
                pruned_data, current_pruning_rate = cdacp.prune_channels_fixed_alpha(down_output, y, self.fixed_alpha)
            # 新增 'random' 选项
            elif self.pruning_tool_name == 'random':
                pruned_data, current_pruning_rate = cdacp.prune_channels_randomly(down_output, y)
            # 新增 'top-k' 选项
            elif self.pruning_tool_name == 'top-k':
                pruned_data, current_pruning_rate = cdacp.prune_top_k(down_output, y)
            else: # 默认使用原版DACP方法
                pruned_data, current_pruning_rate = cdacp.prune_channels(down_output, y)
                self.pruning_rates_history.append(current_pruning_rate)
                output = up_model(pruned_data)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                up_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                up_optimizer.step()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return up_model, self.model, self.pruning_rates_history
    
    def get_train_bath_num(self):
        trainloader = self.load_train_data()
        max_local_epochs = self.local_epochs
        num_batches_per_epoch = len(trainloader)
        return num_batches_per_epoch * max_local_epochs


