import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client

class clientsl(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.trainloader = self.load_train_data()
        self._data_iterator = iter(self.trainloader)
        self.train_cnt=0
    def get_next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            # 尝试从当前迭代器获取下一个批次的数据
            data, labels = next(self._data_iterator)
            return data, labels
        except StopIteration:
            # 如果捕获到 StopIteration 异常，说明当前 epoch 的数据已全部遍历完
            # 重置迭代器，让它从 trainloader 的开头重新提供数据
            self._data_iterator = iter(self.trainloader)
            # 获取新 epoch 的第一个批次数据
            data, labels = next(self._data_iterator)
            return data, labels


    def split_train(self,up_model):
        self.model.train()
        up_model.train() 
        up_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            (x,y)=self.get_next_batch
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            down_output = self.model(x)
            output= up_model(down_output)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            up_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            up_optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


        return up_model,self.model


