import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from torch.utils.data import Dataset, DataLoader


from torch.utils.data import Dataset, DataLoader

class IndexedDataset(Dataset):
    """
    一个包装器，用于处理 [(数据, 标签), ...] 格式的列表，
    并使其返回 (数据, 标签, 索引)。
    """
    def __init__(self, data_list):
        # data_list 是一个形如 [(data1, label1), (data2, label2), ...] 的列表
        self.data_list = data_list

    def __len__(self):
        # 数据集的总长度就是这个列表的长度
        return len(self.data_list)

    def __getitem__(self, index):
        # 根据索引直接从列表中获取 (数据, 标签) 元组
        data, label = self.data_list[index]
        # 返回数据、标签，并附上当前的索引
        return data, label, index



class clientslcs(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs,):
        super().__init__(args, id, train_samples, test_samples,**kwargs)
        self.train_cnt=0
        self.client_data=read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        self.data_select_obj = None
        self.activations_sum={}
        self.Pruning_mata_data=None
        self.hook_layer_name=args.hook_layer_name

    def load_train_index_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.Pruning_mata_data==None:
            train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        else:
            train_data=self.Pruning_mata_data
            
        indexed_train_data = IndexedDataset(train_data)

        return DataLoader(indexed_train_data, batch_size, drop_last=True, shuffle=True)

    def get_activation(self,indices_in_batch,labels_in_batch):

        def hook(model, input, output):

            cloned_output = output.detach().clone().cpu()
            cloned_labels = labels_in_batch.detach().clone().cpu()
            for i, original_idx in enumerate(indices_in_batch):
                self.activations_sum[original_idx.item()] = (cloned_output[i], cloned_labels[i])
        return hook
    
    def split_train(self,up_model):

        self.activations_sum.clear()

        trainloader = self.load_train_index_data()
        
        self.model.train()
        up_model.train() 
        up_optimizer = torch.optim.SGD(up_model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        for epoch in range(max_local_epochs):
            for i, (x, y,index) in enumerate(trainloader):
                handle = getattr(up_model, self.hook_layer_name).register_forward_hook(self.get_activation(index, y))
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
                handle.remove()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # self.data_select_obj=self.args.data_select_obj(self.activations_sum)

        return up_model,self.model
    
    def client_get_data_select(self,data_select_obj):
        if self.data_select_obj == None:
            self.data_select_obj = data_select_obj(self.activations_sum,self.args.data_Pruning_rate)
        else:
            pass


            







