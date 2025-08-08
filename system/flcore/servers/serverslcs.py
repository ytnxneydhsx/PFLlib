import time
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientslcs import clientslcs
import torch.nn as nn
from collections import OrderedDict
import copy
import numpy as np
import logging
from datetime import datetime
import os
from collections import defaultdict



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
        #     client.data_select_obj.select_data_clusters()
        self.current_date=args.current_date
        logger = logging.getLogger(__name__)
        model_res="model_res"
        self.new_dir_path = f"{model_res}/{self.current_date}"
        os.makedirs( self.new_dir_path)
        self.all_centers_list = []
        self.global_centers=None



    def split_evaluate(self,global_model, acc=None, loss=None):
        #这个方法会命令所有客户端用它们各自的测试数据集来评估当前模型，并返回统计结果
        stats = self.test_split_metrics(global_model)
        #这个方法命令所有客户端返回它们在训练数据集上的损失统计信息
        stats_train = self.train_split_metrics(global_model)
        #sum(stats[2]): 所有客户端答对的题目总数。
        #sum(stats[1]): 所有客户端的测试样本总数。
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))    

        logger = logging.getLogger(__name__)
        logger.info("--------------------------------------------------")
        logger.info(f"Averaged Train Loss: {train_loss:.4f}")
        logger.info(f"Averaged Test Accuracy: {test_acc:.4f}")
        logger.info(f"Averaged Test AUC: {test_auc:.4f}")
        logger.info(f"Std Test Accuracy: {np.std(accs):.4f}")
        logger.info(f"Std Test AUC: {np.std(aucs):.4f}")


    def aggregate_and_average_centers(self):
        aggregated_tensors = defaultdict(list)
        for client_center_dict in self.all_centers_list:
            for label, tensor_list in client_center_dict.items():
                aggregated_tensors[label].extend(tensor_list)
        averaged_results = {}
        for label, all_centers_list in aggregated_tensors.items():
            if not all_centers_list:
                continue
            stacked_tensors = torch.stack(all_centers_list, dim=0)
            averaged_results[label] = torch.mean(stacked_tensors, dim=0)
        self.global_centers = averaged_results
        print("全局中心聚合完成！")

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
            #### 全局聚合阶段
            if self.args.data_select_round==0:
                print("开始全局聚合")
                for client in self.selected_clients:
                    client.client_get_data_select(self.data_select_obj)
                    self.all_centers_list.append(client.data_select_obj.center_coordinates)
                self.aggregate_and_average_centers()

                for client in self.selected_clients:
                    client.data_select_obj.global_center_coordinates=self.global_centers
                    client.data_select_obj.Per_Pruning_data()
                    client.Pruning_mata_data=client.data_select_obj.Pruning_data(client.client_data)
                

                    
                    
                    



            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            self.global_model = nn.Sequential(
            copy.deepcopy(self.down_model),
            copy.deepcopy(self.up_model)
            )
            # 2. 创建一个新的状态字典，移除 nn.Sequential 的前缀
            new_state_dict = {}
            for key, value in self.global_model.state_dict().items():
                # 键名示例：'0.conv1.0.weight'
                # 我们需要将其转换为 'conv1.0.weight'
                # 寻找第一个 '.'
                parts = key.split('.', 1)
                if len(parts) > 1:
                    new_key = parts[1]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            if i%1 == 0:
                print("当前轮次为"+str(i))
                model_name = f"{self.args.algorithm}_{self.args.dataset}_{self.args.model_str}_{i}.pt"
                model_name = os.path.join( self.new_dir_path, model_name)
                torch.save(new_state_dict, model_name)
                print(f"模型已保存为: {model_name}") 
                logger = logging.getLogger(__name__)
                logger.info(f"模型已保存为: {model_name}")
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        # self.save_results()
        # self.save_global_model()
