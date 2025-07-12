import time
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientsl import clientsl
import torch.nn as nn
from collections import OrderedDict
import copy

class sl(Server):
    def __init__(self, args, times,Split_cnt):
        super().__init__(args,times)
         # select slow clients
        self.set_slow_clients()
        (self.down_model,self.up_model)=self.split_model(self.global_model,Split_cnt)
        self.Split_cnt=Split_cnt
        self.set_split_clients(clientsl,self.down_model)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
    
    def split_model(self,model: nn.Module, split_point: int) -> tuple[nn.Module, nn.Module]:
        """
        将给定的 nn.Module 模型切分为两个 nn.Sequential 模块。

        参数:
            model (nn.Module): 要切分的模型。
            split_point (int): 分割点，表示在模型的第 `split_point` 层之后进行切分。
                                （从0开始计数，split_point 层包含在第一个返回的模型中）。

        返回:
            tuple[nn.Module, nn.Module]: 一个包含两个 nn.Sequential 模块的元组 (model_part1, model_part2)。
                                        如果 split_point 超出范围，将抛出 ValueError 或 IndexError。
        """
        all_layers = OrderedDict(model.named_children())
        
        if not all_layers:
            raise ValueError("模型没有可切分的子层。")
        # split_point 应该在 [0, len(all_layers) - 1] 范围内
        if split_point < 0 or split_point >= len(all_layers): 
            raise IndexError(f"切分点 {split_point} 超出模型层数范围 ({len(all_layers)} 层)。有效范围是 [0, {len(all_layers)-1}]。")

        layers_part1 = OrderedDict()
        layers_part2 = OrderedDict()

        for i, (name, layer) in enumerate(all_layers.items()):
            if i <= split_point: # 包含 split_point 对应的层
                layers_part1[name] = layer
            else:
                layers_part2[name] = layer

        model_part1 = nn.Sequential(layers_part1)
        model_part2 = nn.Sequential(layers_part2)

        return model_part1, model_part2
    
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
                
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))



        self.save_results()
        self.save_global_model()


                    
                    














