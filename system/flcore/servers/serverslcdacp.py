import time
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientslcdacp import clientslcdacp
import logging
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from datetime import datetime
from collections import defaultdict
import copy
import os
from channelstools.channelstoolslcdacp import channelstoolcdacp

class slcdacp(Server):
    def __init__(self, args, times, Split_cnt):
        super().__init__(args, times)
        self.set_slow_clients()
        (self.down_model, self.up_model) = self.split_model(self.global_model, Split_cnt)
        self.Split_cnt = Split_cnt
        self.sum_bath_cnt = 0

        self.set_split_clients(clientslcdacp, self.down_model)
        self.selected_clients = self.select_clients()
        
        for client in self.selected_clients:
            self.sum_bath_cnt = client.get_train_bath_num() + self.sum_bath_cnt
        self.sum_bath_cnt = self.sum_bath_cnt * self.args.global_rounds
        self.cdacp = channelstoolcdacp(self.args, self.sum_bath_cnt)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.current_date = args.current_date
        logger = logging.getLogger(__name__)
        model_res = "model_res"
        self.new_dir_path = f"{model_res}/{args.algorithm}_{args.model_str}_{args.dataset}_{args.current_date}_{self.current_date}"
        os.makedirs(self.new_dir_path, exist_ok=True)
        self.all_centers_list = []
        self.global_centers = None

    def split_evaluate(self, global_model, acc=None, loss=None):
        stats = self.test_split_metrics(global_model)
        stats_train = self.train_split_metrics(global_model)
        
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1]) if sum(stats_train[1]) > 0 else 0
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        logger = logging.getLogger(__name__)
        logger.info("--------------------------------------------------")
        logger.info(f"Averaged Train Loss: {train_loss:.4f}")
        logger.info(f"Averaged Test Accuracy: {test_acc:.4f}")
        logger.info(f"Averaged Test AUC: {test_auc:.4f}")
        logger.info(f"Std Test Accuracy: {np.std(accs):.4f}")
        logger.info(f"Std Test AUC: {np.std(aucs):.4f}")

    def train(self):
        for i in range(self.global_rounds + 1):
            logger = logging.getLogger(__name__)
            s_t = time.time()
            client_pruning_rates = {}
            self.send_split_models(self.down_model)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.split_evaluate(self.global_model)

            for client in self.selected_clients:
                self.up_model, self.down_model, rates = client.split_train(self.up_model, self.cdacp)
                self.send_split_models(self.down_model)
                if rates:
                    avg_rate = sum(rates) / len(rates)
                    client_pruning_rates[client.id] = avg_rate

            if client_pruning_rates:
                avg_rate_for_round = sum(client_pruning_rates.values()) / len(client_pruning_rates)
                print(f"-------------Round {i} Pruning Rates-------------")
                logger.info(f"-------------Round {i} Pruning Rates-------------")
                for client_id, avg_rate in client_pruning_rates.items():
                    print(f"Client {client_id} Average Pruning Rate: {avg_rate:.4f}")
                    logger.info(f"Client {client_id} Average Pruning Rate: {avg_rate:.4f}")
                print(f"Overall Average Pruning Rate for Round {i}: {avg_rate_for_round:.4f}")
                logger.info(f"Overall Average Pruning Rate for Round {i}: {avg_rate_for_round:.4f}")

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            
            self.global_model = nn.Sequential(copy.deepcopy(self.down_model), copy.deepcopy(self.up_model))

    def _evaluate_model_for_round(self, model_to_evaluate, clients_to_test):
        num_samples_test, tot_correct_test = [], []
        for client in clients_to_test:
            ct, ns, _ = client.test_split_metrics(model_to_evaluate)
            tot_correct_test.append(ct)
            num_samples_test.append(ns)
        test_acc = sum(tot_correct_test) * 1.0 / sum(num_samples_test) if sum(num_samples_test) > 0 else 0

        num_samples_train, train_losses = [], []
        for client in clients_to_test:
            tl, ns = client.train_split_metrics(model_to_evaluate)
            train_losses.append(tl * ns)
            num_samples_train.append(ns)
        train_loss = sum(train_losses) / sum(num_samples_train) if sum(num_samples_train) > 0 else 0
        
        return test_acc, train_loss

    def train_staged_alpha_experiment(self):
        print("Starting Staged Optimal Alpha Investigation Experiment...")
        logger = logging.getLogger(__name__)
        stage_length = self.args.stage_length
        alpha_values_to_test = [float(x) for x in self.args.alpha_test_values.split(',')]
        num_stages = self.global_rounds // stage_length
        optimal_alphas = {}
        best_up_model_state = copy.deepcopy(self.up_model.state_dict())
        best_down_model_state = copy.deepcopy(self.down_model.state_dict())

        for stage_idx in range(num_stages):
            stage_start_round = stage_idx * stage_length
            stage_end_round = (stage_idx + 1) * stage_length
            print(f"\n{'='*20} STAGE {stage_idx + 1} (Rounds {stage_start_round+1}-{stage_end_round}) {'='*20}")
            logger.info(f"===== STAGE {stage_idx + 1} (Rounds {stage_start_round+1}-{stage_end_round}) =====")
            stage_results = {}

            for alpha in alpha_values_to_test:
                print(f"\n--- Testing Alpha = {alpha:.2f} for Stage {stage_idx + 1} ---")
                logger.info(f"--- Testing Alpha = {alpha:.2f} for Stage {stage_idx + 1} ---") # Log header for alpha
                
                sim_up_model = copy.deepcopy(self.up_model)
                sim_down_model = copy.deepcopy(self.down_model)
                sim_up_model.load_state_dict(best_up_model_state)
                sim_down_model.load_state_dict(best_down_model_state)
                sim_cdacp = channelstoolcdacp(self.args, self.sum_bath_cnt)
                sim_clients = []
                for client_template in self.clients:
                    sim_client = clientslcdacp(
                        self.args, client_template.id, client_template.train_samples, 
                        client_template.test_samples,
                        down_model=copy.deepcopy(sim_down_model),
                        train_slow=client_template.train_slow,
                        send_slow=client_template.send_slow
                    )
                    sim_client.pruning_tool_name = 'fixed_alpha'
                    sim_client.fixed_alpha = alpha
                    sim_clients.append(sim_client)
                
                # Still need history in memory to find the best final accuracy
                round_history = [] 
                
                for r in range(stage_start_round, stage_end_round):
                    print(f"\rAlpha {alpha:.2f} | Training Round {r+1}/{self.global_rounds}", end="")
                    for client in sim_clients:
                        sim_up_model, updated_down_model, _ = client.split_train(sim_up_model, sim_cdacp)
                        sim_down_model.load_state_dict(updated_down_model.state_dict())
                        for other_client in sim_clients:
                            other_client.model.load_state_dict(sim_down_model.state_dict())
                    
                    eval_model = nn.Sequential(copy.deepcopy(sim_down_model), copy.deepcopy(sim_up_model))
                    test_acc_round, train_loss_round = self._evaluate_model_for_round(eval_model, sim_clients)
                    
                    # MODIFICATION: Log to file immediately after each round's evaluation
                    log_message = f"Stage {stage_idx + 1}, Alpha {alpha:.2f}, Round {r + 1}: Accuracy={test_acc_round:.4f}, Loss={train_loss_round:.4f}"
                    logger.info(log_message)

                    # Append to in-memory history to find the final accuracy for this stage
                    round_history.append({'accuracy': test_acc_round})

                final_acc_for_stage = round_history[-1]['accuracy'] if round_history else 0
                print(f"\nResult for Alpha = {alpha:.2f}: Final Accuracy = {final_acc_for_stage:.4f}")
                
                stage_results[alpha] = {
                    'accuracy': final_acc_for_stage,
                    'up_model_state': copy.deepcopy(sim_up_model.state_dict()),
                    'down_model_state': copy.deepcopy(sim_down_model.state_dict()),
                }

            if not stage_results:
                print("No results to process for this stage. Stopping.")
                return

            best_alpha_for_stage = max(stage_results, key=lambda k: stage_results[k]['accuracy'])
            best_result = stage_results[best_alpha_for_stage]
            optimal_alphas[f"Stage {stage_idx + 1} (Rounds {stage_start_round+1}-{stage_end_round})"] = best_alpha_for_stage
            best_up_model_state = best_result['up_model_state']
            best_down_model_state = best_result['down_model_state']
            
            # Announce the winner for the stage
            print(f"\n** Best Alpha for Stage {stage_idx + 1} is {best_alpha_for_stage:.2f} with Accuracy {best_result['accuracy']:.4f} **")
            logger.info(f"** Best Alpha for Stage {stage_idx + 1}: {best_alpha_for_stage:.2f} with Accuracy: {best_result['accuracy']:.4f} **")

        print("\n\n{'='*25} EXPERIMENT SUMMARY {'='*25}")
        print(f"| {'Training Stage (Rounds)':<30} | {'Optimal α Value':<15} |")
        print(f"| :{'-'*29} | :{'-'*14} |")
        for stage_name, alpha_val in optimal_alphas.items():
            print(f"| {stage_name:<30} | {alpha_val:<15.2f} |")
        print("=" * 52)
        logger.info("===== EXPERIMENT SUMMARY =====")
        for stage_name, alpha_val in optimal_alphas.items():
            logger.info(f"Stage: {stage_name}, Optimal Alpha: {alpha_val:.2f}")