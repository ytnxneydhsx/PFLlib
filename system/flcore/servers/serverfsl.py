import time
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientfsl import clientfsl
import torch.nn as nn
import copy

class fsl(Server):
    def __init__(self, args, times,Split_cnt):
        super().__init__(args,times)
         # select slow clients
        self.set_slow_clients()
        (self.down_model,self.up_model)=self.split_model(self.global_model,Split_cnt)
        self.Split_cnt=Split_cnt
        self.set_split_clients(clientfsl,self.down_model)
        (self.down_model_list,self.up_model_list)=self.creat_model_list()


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
    
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_split_models(self.down_model)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.split_evaluate(self.global_model)
            
            for i,client in enumerate(self.selected_clients):
                # self.up_model_list[i].train()
                # up_optimizer = torch.optim.SGD(self.up_model_list[i].parameters(), lr=self.learning_rate)
                # up_optimizer.zero_grad()
                (self.up_model_list[i],self.down_model_list[i])=client.split_train(self.up_model_list[i])
                # up_optimizer.step()

            self.receive_models()

            #聚合模型
            self.up_model=self.aggregate_model(self.up_model_list)
            self.down_model=self.aggregate_model(self.down_model_list)

            self.up_model_list=[self.up_model]*self.num_join_clients
            
            #本轮花费时间
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            self.global_model = nn.Sequential(
            copy.deepcopy(self.down_model),
            copy.deepcopy(self.up_model)
            )
                
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))



        self.save_results()
        self.save_global_model()

    def creat_model_list(self):
        up_model_list=[]
        down_model_list=[]
        for i in range(self.num_join_clients):
            up_model_list.append(self.up_model)
            down_model_list.append(self.down_model)
        return down_model_list,up_model_list
    
    def aggregate_model(self,mode_list):
        global_model = copy.deepcopy(mode_list[0])
        for param in global_model.parameters():
            param.data.zero_()
        for w, model in zip(self.uploaded_weights, mode_list):
            for server_param, client_param in zip(global_model.parameters(), model.parameters()):
                server_param.data += client_param.data.clone() * w
        return global_model

        

        




    