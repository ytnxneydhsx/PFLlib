import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data

class clientslcs(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs,):
        super().__init__(args, id, train_samples, test_samples,**kwargs)
        self.train_cnt=0
        self.data_select_obj=None
        self.client_data=read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)

        

    def split_train(self,up_model):

        trainloader = self.load_train_data()
        
        self.model.train()
        up_model.train() 
        up_optimizer = torch.optim.SGD(up_model.parameters(), lr=self.learning_rate)
        
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
    
    def client_data_select(self):
        if self.data_select_obj == None:
            pass
        else:
            self.client_data = self.data_select_obj.select_data_sort()
            







