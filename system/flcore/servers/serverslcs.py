import time
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientslcs import clientslcs
import torch.nn as nn
from collections import OrderedDict
import copy



class slcs(Server):
    def __init__(self, args, times,Split_cnt):
        super().__init__(args,times)
         # select slow clients
        self.set_slow_clients()
        (self.down_model,self.up_model)=self.split_model(self.global_model,Split_cnt)
        self.Split_cnt=Split_cnt
        self.set_split_clients(clientslcs,self.down_model)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        # self.load_model()
        self.Budget = []
        # for client in self.clients:
        #     client.data_select_obj=args.data_select_obj(client.client_data)
            # client.data_select_obj.select_data_clusters()





    

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_split_models(self.down_model)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.split_evaluate(self.global_model)
            for client in self.selected_clients:
                (self.up_model,self.down_model)=client.split_train(self.up_model)
                self.send_split_models(self.down_model)
            #本轮花费时间
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            self.global_model = nn.Sequential(
            copy.deepcopy(self.down_model),
            copy.deepcopy(self.up_model)
            )
            if i%10 == 0:
                model_name = f"ResNet_SL_round_{i}.pt"
                torch.save(self.global_model.state_dict(), model_name)
                print(f"模型已保存为: {model_name}") 

            
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


        # self.save_results()
        # self.save_global_model()
