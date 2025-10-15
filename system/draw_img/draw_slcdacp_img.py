import os
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from drawbase import drawbase
import sys
import shutil
import ast  # 用于安全地将字符串转为字典

# 设置Matplotlib后端，防止在无图形界面的环境中报错
matplotlib.use('Agg')

class draw_slcdacp(drawbase):
    def __init__(self, log_filename=None):
        super().__init__(log_filename)
        self.log_filename = log_filename
        self.algorithm = None
        self.model_str = None
        self.dataset = None
        self.current_date = None
        self.local_learning_rate = None
        self.purning_min = None
        self.purning_base = None
        self.purning_max = None
        self.prune_tool = None
        self.split_model_cnt = None
        self.fixed_alpha = None
        self.batch_size = None
        self.global_rounds = None
        self.optimizer_str = None
        self.momentum= None
        self.alpha = None # 为兼容性添加 alpha 属性

        self.channel_increases = {}
        # --- 新增：为通道分数数据初始化字典 ---
        self.historical_scores = {}
        self.this_round_scores = {}
        # ------------------------------------

        self._load_parameters_from_logger()

        # 完全遵循您原有的 output_dir 构建逻辑
        if self.algorithm and self.model_str and self.dataset and self.current_date:
            self.base_name = f"{self.algorithm}_{self.model_str}_{self.dataset}_{self.current_date}"
            alpha_str = f"alpha_{self.alpha}" if self.alpha is not None else "alpha_N-A"
            lr_str = f"lr_{self.local_learning_rate}" if self.local_learning_rate is not None else "lr_N-A"
            time_str = f"time_{self.current_date}" if self.current_date is not None else "time_N-A"
            purning_min_str = f"purning_min_{self.purning_min}" if self.purning_min is not None else "purning_min_N-A"
            purning_base_str = f"purning_base_{self.purning_base}" if self.purning_base is not None else "purning_base_N-A"
            purning_max_str = f"purning_max_{self.purning_max}" if self.purning_max is not None else "purning_max_N-A"
            prune_tool_str = f"{self.prune_tool}" if self.prune_tool is not None else "default"
            batch_size_str = f"batch_size_{self.batch_size}" if self.batch_size is not None else "batch_size_N-A"
            optimizer_str = f"optimizer_{self.optimizer_str}" if self.optimizer_str is not None else "SGD"
            global_rounds_str = f"global_rounds_{self.global_rounds}" if self.global_rounds is not None else "global_rounds_N-A"
            split_model_cnt_str = f"split_model_cnt_{self.split_model_cnt}" if self.split_model_cnt is not None else "split_model_cnt_N-A"
            path_components = [
                self.root_path, self.algorithm, self.model_str, self.dataset,
                alpha_str, prune_tool_str, purning_min_str, purning_base_str,
                purning_max_str, batch_size_str, lr_str, optimizer_str, global_rounds_str,
                split_model_cnt_str,
            ]
            if prune_tool_str == 'fixed_alpha' and self.fixed_alpha is not None:
                path_components.extend(str(self.fixed_alpha).split('/'))
            path_components.append(time_str)
            self.output_dir = os.path.join(*path_components)
            self.output_dir = self.output_dir.replace(':', '_')
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None

    def _load_parameters_from_logger(self):
        if not self.log_file_path or not os.path.exists(self.log_file_path): return
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f: content = f.read()
            params = {
                'algorithm': r'algorithm\s*=\s*(\w+)',
                'model_str': r'model\s*=\s*(\w+)',
                'dataset': r'dataset\s*=\s*(\w+)',
                'current_date': r'current_date\s*=\s*(\S+)',
                'alpha': r'alpha\s*=\s*(\d+\.?\d*)', 
                'batch_size': r'batch_size\s*=\s*(\d+)',
                'local_learning_rate': r'local_learning_rate\s*=\s*(\d+\.?\d*)',
                'purning_min': r'purning_min\s*=\s*(\d+\.?\d*)', 
                'purning_base': r'purning_base\s*=\s*(\d+\.?\d*)',
                'purning_max': r'purning_max\s*=\s*(\d+\.?\d*)', 
                'split_model_cnt': r'split_model_cnt\s*=\s*(\d+)',
                'prune_tool': r'prune_tool\s*=\s*([\w-]+)', 
                'fixed_alpha': r'fixed_alpha\s*=\s*([\w./-]+)',
                'global_rounds': r'global_rounds\s*=\s*(\d+\.?\d*)',
                'optimizer_str': r'optimizer_str\s*=\s*(\w+)',
            }
            for key, pattern in params.items():
                match = re.search(pattern, content)
                if match:
                    val = match.group(1).strip()
                    if key in ['fixed_alpha', 'prune_tool', 'optimizer_str', 'algorithm', 'model_str', 'dataset', 'current_date']: 
                        setattr(self, key, val)
                    elif key in ['alpha', 'local_learning_rate', 'purning_min', 'purning_base', 'purning_max', 'global_rounds']: 
                        setattr(self, key, float(val))
                    else: 
                        setattr(self, key, int(val))
        except Exception as e: print(f"Error parsing log file: {e}")

    def plot_acc_img(self):
        if not self.log_filename or not self.output_dir: return
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        accuracies = []
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Averaged Test Accuracy:" in line:
                        match = re.search(r"(\d+\.\d+)", line)
                        if match: accuracies.append(float(match.group(1)))
        except FileNotFoundError: return
        if not accuracies: return
        plt.figure(figsize=(12, 7)); plt.plot(accuracies, linestyle='-', color='b')
        plt.title(f"{self.base_name} - Accuracy"); plt.xlabel('Epoch'); plt.ylabel('Averaged Test Accuracy')
        plt.grid(True); plt.tight_layout(); plt.savefig(full_output_path)
        print(f"Accuracy chart saved: '{full_output_path}'")
        plt.close()

    def plot_loss_img(self):
        if not self.log_filename or not self.output_dir: return
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}_loss.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        losses = []
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Averaged Train Loss:" in line:
                        match = re.search(r"(\d+\.\d+)", line)
                        if match: losses.append(float(match.group(1)))
        except FileNotFoundError: return
        if not losses: return
        plt.figure(figsize=(12, 7)); plt.plot(losses, linestyle='--', color='r')
        plt.title(f"{self.base_name} - Loss"); plt.xlabel('Epoch'); plt.ylabel('Averaged Train Loss')
        plt.grid(True); plt.tight_layout(); plt.savefig(full_output_path)
        print(f"Loss chart saved: '{full_output_path}'")
        plt.close()

    def plot_pruning_rate_img(self):
        if not self.log_filename or not self.output_dir: return
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}_pruning_rate.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        per_round_rates = []
        pattern = re.compile(r"Overall Average Pruning Rate for Round \d+: (\d+\.\d+)")
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                per_round_rates = [float(rate) for rate in pattern.findall(f.read())]
        except FileNotFoundError: return
        if not per_round_rates: return
        cumulative = [sum(per_round_rates[:i+1]) / (i + 1) for i in range(len(per_round_rates))]
        global_avg = sum(per_round_rates) / len(per_round_rates)
        global_line = [global_avg] * len(per_round_rates)
        plt.figure(figsize=(12, 7))
        plt.plot(per_round_rates, linestyle='-', color='g', label='Current Round Avg')
        plt.plot(cumulative, linestyle='--', color='r', label='Cumulative Avg')
        plt.plot(global_line, linestyle=':', color='b', label=f'Global Avg: {global_avg:.4f}')
        plt.title(f"{self.base_name} - Pruning Rates"); plt.xlabel('Epoch'); plt.ylabel('Average Pruning Rate')
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(full_output_path)
        print(f"Pruning rate chart saved: '{full_output_path}'")
        plt.close()

    def get_channel_increases(self):
        """解析日志文件以提取通道增加值数据。"""
        self.channel_increases.clear()
        if not self.log_file_path or not os.path.exists(self.log_file_path): return
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            current_round = 0
            pattern = re.compile(r"Increments for all channels this round: (\{.*\})")
            
            for line in f:
                round_match = re.search(r'Round\s+(\d+)', line)
                if round_match:
                    current_round = int(round_match.group(1))

                match = pattern.search(line)
                if match:
                    dict_str = match.group(1)
                    client_id = 0 
                    layer_name = 'aggregated_layer'
                    try:
                        increase_dict = ast.literal_eval(dict_str)
                        self.channel_increases.setdefault(layer_name, {}).setdefault(client_id, {})[current_round] = increase_dict
                    except (ValueError, SyntaxError) as e:
                        print(f"无法解析第 {current_round} 轮的字典: {dict_str}，错误: {e}")

    def plot_channel_heatmap(self):
        """为通道增加值绘制并保存热力图。"""
        if not self.channel_increases or not self.output_dir:
            print(f"没有为 {self.log_filename} 找到用于绘制热力图的数据。")
            return
        for layer_name, clients_data in self.channel_increases.items():
            for client_id, rounds_data in clients_data.items():
                if not rounds_data: continue
                df = pd.DataFrame.from_dict(rounds_data, orient='index').sort_index().transpose().sort_index().fillna(0)
                if df.empty:
                    print(f"数据帧为空，跳过热力图绘制: Layer {layer_name}, Client {client_id}"); continue
                
                plt.figure(figsize=(20, 10))
                sns.heatmap(df, cmap='viridis', cbar=True)
                plt.title(f'通道增加值热力图\n{self.base_name}\nLayer: {layer_name} - Client: {client_id}')
                plt.xlabel('轮次 (Round)'); plt.ylabel('通道索引 (Channel Index)')
                
                heatmap_filename = os.path.join(self.output_dir, f"{self.base_name}_heatmap_layer_{layer_name.replace('.', '_')}_client_{client_id}.png")
                plt.savefig(heatmap_filename, bbox_inches='tight'); plt.close()
                print(f"热力图已保存: {heatmap_filename}")

    # --- 以下是新增的函数 ---
    def get_channel_scores(self):
        """解析日志文件，提取历史平均分数和当前轮次分数。"""
        self.historical_scores.clear()
        self.this_round_scores.clear()
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return

        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            current_round = -1
            # 为两种分数设置正则表达式
            historical_pattern = re.compile(r"Historical Averages: (\{.*\})")
            this_round_pattern = re.compile(r"This Round's Scores: (\{.*\})")
            
            for line in f:
                round_match = re.search(r'--- Channel Score Analysis at the end of Round (\d+) ---', line)
                if round_match:
                    current_round = int(round_match.group(1))

                # 匹配历史平均分数
                historical_match = historical_pattern.search(line)
                if historical_match and current_round != -1:
                    dict_str = historical_match.group(1)
                    try:
                        scores_dict = ast.literal_eval(dict_str)
                        # 使用默认的层和客户端ID进行存储
                        self.historical_scores.setdefault('aggregated_layer', {}).setdefault(0, {})[current_round] = scores_dict
                    except (ValueError, SyntaxError) as e:
                        print(f"无法解析第 {current_round} 轮的历史平均分数: {dict_str}, 错误: {e}")

                # 匹配当前轮次分数
                this_round_match = this_round_pattern.search(line)
                if this_round_match and current_round != -1:
                    dict_str = this_round_match.group(1)
                    try:
                        scores_dict = ast.literal_eval(dict_str)
                        # 使用默认的层和客户端ID进行存储
                        self.this_round_scores.setdefault('aggregated_layer', {}).setdefault(0, {})[current_round] = scores_dict
                    except (ValueError, SyntaxError) as e:
                        print(f"无法解析第 {current_round} 轮的当前轮次分数: {dict_str}, 错误: {e}")

    def plot_historical_scores_heatmap(self):
        """为历史平均通道分数绘制并保存热力图。"""
        if not self.historical_scores or not self.output_dir:
            return
        for layer_name, clients_data in self.historical_scores.items():
            for client_id, rounds_data in clients_data.items():
                if not rounds_data: continue
                df = pd.DataFrame.from_dict(rounds_data, orient='index').sort_index().transpose().sort_index().fillna(0)
                if df.empty: continue
                
                plt.figure(figsize=(20, 10))
                sns.heatmap(df, cmap='viridis', cbar=True)
                plt.title(f'历史平均通道分数热力图\n{self.base_name}\nLayer: {layer_name} - Client: {client_id}')
                plt.xlabel('轮次 (Round)'); plt.ylabel('通道索引 (Channel Index)')
                
                heatmap_filename = os.path.join(self.output_dir, f"{self.base_name}_heatmap_historical_scores.png")
                plt.savefig(heatmap_filename, bbox_inches='tight'); plt.close()
                print(f"历史分数热力图已保存: {heatmap_filename}")

    def plot_this_round_scores_heatmap(self):
        """为当前轮次的通道分数绘制并保存热力图。"""
        if not self.this_round_scores or not self.output_dir:
            return
        for layer_name, clients_data in self.this_round_scores.items():
            for client_id, rounds_data in clients_data.items():
                if not rounds_data: continue
                df = pd.DataFrame.from_dict(rounds_data, orient='index').sort_index().transpose().sort_index().fillna(0)
                if df.empty: continue

                plt.figure(figsize=(20, 10))
                # 使用 RdBu_r 色彩映射，正值为红，负值为蓝，中心为白，可以更好地观察分数波动
                sns.heatmap(df, cmap='RdBu_r', center=0, cbar=True)
                plt.title(f'当前轮次通道分数热力图\n{self.base_name}\nLayer: {layer_name} - Client: {client_id}')
                plt.xlabel('轮次 (Round)'); plt.ylabel('通道索引 (Channel Index)')
                
                heatmap_filename = os.path.join(self.output_dir, f"{self.base_name}_heatmap_this_round_scores.png")
                plt.savefig(heatmap_filename, bbox_inches='tight'); plt.close()
                print(f"当前轮次分数热力图已保存: {heatmap_filename}")
    # --- 新增功能结束 ---

    def draw_acc_and_place_in_folder(self):
        print("--- 开始扫描并处理日志文件 ---")
        if not os.path.exists(self.root_path): return
        for filename in os.listdir(self.root_path):
            if filename.endswith(".log"):
                try:
                    print(f"\n>>> 正在处理文件: {filename}")
                    drawer = draw_slcdacp(filename)
                    if not drawer.output_dir:
                        print(f"无法确定输出目录，跳过文件 '{filename}'")
                        continue
                    drawer.plot_acc_img()
                    drawer.plot_loss_img()
                    drawer.plot_pruning_rate_img()

                    drawer.get_channel_increases()
                    drawer.plot_channel_heatmap()

                    # --- 新增的函数调用 ---
                    drawer.get_channel_scores()
                    drawer.plot_historical_scores_heatmap()
                    drawer.plot_this_round_scores_heatmap()
                    # ----------------------

                    log_path = os.path.join(self.root_path, filename)
                    new_log_path = os.path.join(drawer.output_dir, filename)
                    if os.path.exists(log_path):
                        shutil.move(log_path, new_log_path)
                        print(f"日志文件已移动到: '{new_log_path}'")
                except Exception as e:
                    print(f"处理 '{filename}' 时出错: {e}")
        print("\n--- 所有文件处理完毕 ---")
    
    def flatten_logs_and_cleanup_images(self):
        print("--- 开始预处理任务 ---")
        if not os.path.exists(self.root_path): return
        for root, _, files in os.walk(self.root_path):
            if root == self.root_path: continue
            for file in files:
                if file.endswith(".log"):
                    try: shutil.move(os.path.join(root, file), os.path.join(self.root_path, file))
                    except Exception as e: print(f"移动 '{file}' 时出错: {e}")
        for filename in os.listdir(self.root_path):
            file_path = os.path.join(self.root_path, filename)
            if filename.lower().endswith((".jpg", ".png")):
                try: os.remove(file_path)
                except Exception as e: print(f"删除 '{file_path}' 时出错: {e}")
            elif os.path.isdir(file_path):
                try: shutil.rmtree(file_path)
                except Exception as e: print(f"删除文件夹 '{file_path}' 时出错: {e}")
        print("--- 预处理完成 ---")


if __name__ == '__main__':
    # 您的原始 main 结构保持不变
    processor = draw_slcdacp()
    # To run cleanup, uncomment the following line
    # processor.flatten_logs_and_cleanup_images() 
    processor.draw_acc_and_place_in_folder()