import os
import re
import matplotlib.pyplot as plt
import matplotlib
from drawbase import drawbase  # 假设 drawbase.py 文件存在且定义正确
import sys
import shutil

# 设置matplotlib后端，防止在无GUI环境下报错
matplotlib.use('Agg')
# 获取当前工作目录
current_dir = os.getcwd()
sys.path.append(current_dir)

class draw_slcdacp(drawbase):
    def __init__(self, log_filename=None):
        """
        构造函数，用于初始化对象并从日志文件中解析参数。
        """
        super().__init__(log_filename)
        self.log_filename = log_filename
        self.local_learning_rate = None
        self.purning_min = None
        self.purning_base = None
        self.purning_max = None
        self.prune_tool = None
        self.split_model_cnt = None
        self.fixed_alpha = None
        self._load_parameters_from_logger()

        # 仅当成功解析到基础属性时才构建目录
        if self.base_name and self.algorithm and self.model_str and self.dataset:
            alpha_str = f"alpha_{self.alpha}" if self.alpha is not None else "alpha_N-A"
            lr_str = f"lr_{self.local_learning_rate}" if self.local_learning_rate is not None else "lr_N-A"
            time_str = f"time_{self.time}" if self.time is not None else "time_N-A"
            purning_min_str = f"purning_min_{self.purning_min}" if self.purning_min is not None else "purning_min_N-A"
            purning_base_str = f"purning_base_{self.purning_base}" if self.purning_base is not None else "purning_base_N-A"
            purning_max_str = f"purning_max_{self.purning_max}" if self.purning_max is not None else "purning_max_N-A"
            prune_tool_str = f"{self.prune_tool}" if self.prune_tool is not None else "default"
            if prune_tool_str == "defult":
                prune_tool_str = "default"
            if prune_tool_str == "defult_recent_10":
                prune_tool_str = "default_recent_10"
            split_model_cnt_str = f"split_model_cnt_{self.split_model_cnt}" if self.split_model_cnt is not None else "split_model_cnt_2"

            # 【解释】: 我们创建一个列表，按顺序放入所有目录名
            path_components = [
                self.root_path,
                self.algorithm,
                self.model_str,
                self.dataset,
                alpha_str,
                prune_tool_str,
                purning_min_str,
                purning_base_str,
                purning_max_str,
                lr_str,               # lr_str 也包含在内
                split_model_cnt_str,  # 这是插入位置之前的最后一个固定目录
            ]
            if prune_tool_str == 'fixed_alpha' and self.fixed_alpha is not None:
                fixed_alpha_folders = str(self.fixed_alpha).split('/')
                path_components.extend(fixed_alpha_folders)
            path_components.append(time_str)
            self.output_dir = os.path.join(*path_components)
            
            self.output_dir = self.output_dir.replace(':', '_')
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"输出目录已准备好: '{self.output_dir}'")
        else:
            self.output_dir = None
            print(f"警告：由于文件名未提供或格式不正确，无法创建输出目录。")

    def plot_acc_img(self):
        """
        绘制并保存在测试集上的平均准确率图表。
        """
        if not self.log_filename or not self.output_dir:
            print("错误：未指定日志文件或输出目录，无法绘制图表。")
            return
        
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        print(f"正在从 '{full_log_path}' 读取数据并生成准确率图表...")
        
        accuracies = []
        accuracy_pattern = re.compile(r"Averaged Test Accuracy: (\d+\.\d+)")
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = accuracy_pattern.search(line)
                    if match:
                        accuracies.append(float(match.group(1)))
        except FileNotFoundError:
            print(f"错误：文件 '{full_log_path}' 未找到。")
            return
        except Exception as e:
            print(f"发生错误: {e}")
            return
        
        if not accuracies:
            print("在日志文件中未找到准确率数据。")
            return
        
        plt.figure(figsize=(12, 7))
        plt.plot(accuracies, marker='o', linestyle='-', color='b')
        
        title_str = f"{self.base_name}"
        if self.alpha is not None:
            title_str += f"_alpha:{self.alpha}"
        if self.purning_min is not None:
            title_str += f" purning_min:{self.purning_min}"
        if self.purning_base is not None:
            title_str += f" purning_base:{self.purning_base}"
        if self.purning_max is not None:
            title_str += f" purning_max:{self.purning_max}"
        if self.prune_tool is not None:
            title_str += f" prune_tool:{self.prune_tool}"
        if self.algorithm == 'fixed_alpha' and self.fixed_alpha is not None:
            title_str += f" fixed_alpha:{self.fixed_alpha}"
        if self.split_model_cnt is not None:
            title_str += f" split_model_cnt:{self.split_model_cnt}"
        plt.title(title_str)
        
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('平均测试准确率 (Averaged Test Accuracy)')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        plt.savefig(full_output_path)
        print(f"准确率图表已成功生成并保存为: '{full_output_path}'")
        plt.close()

    def plot_loss_img(self):
        """
        绘制并保存在训练集上的平均损失图表。
        """
        if not self.log_filename or not self.output_dir:
            print("错误：未指定日志文件或输出目录，无法绘制损失图表。")
            return
        
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}_loss.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        print(f"正在从 '{full_log_path}' 读取数据并生成损失图表...")
        
        losses = []
        loss_pattern = re.compile(r"Averaged Train Loss: (\d+\.\d+)")
        
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = loss_pattern.search(line)
                    if match:
                        losses.append(float(match.group(1)))
        except FileNotFoundError:
            print(f"错误：文件 '{full_log_path}' 未找到。")
            return
        except Exception as e:
            print(f"发生错误: {e}")
            return
        
        if not losses:
            print("在日志文件中未找到损失数据。")
            return
        
        plt.figure(figsize=(12, 7))
        plt.plot(losses, marker='x', linestyle='--', color='r')
        
        title_str = f"{self.base_name} - Loss"
        if self.alpha is not None:
            title_str += f"_alpha:{self.alpha}"
        if self.purning_min is not None:
            title_str += f" purning_min:{self.purning_min}"
        if self.purning_base is not None:
            title_str += f" purning_base:{self.purning_base}"
        if self.purning_max is not None:
            title_str += f" purning_max:{self.purning_max}"
        if self.prune_tool is not None:
            title_str += f" prune_tool:{self.prune_tool}"
        if self.algorithm == 'fixed_alpha' and self.fixed_alpha is not None:
            title_str += f" fixed_alpha:{self.fixed_alpha}"
        if self.split_model_cnt is not None:
            title_str += f" split_model_cnt:{self.split_model_cnt}"
        plt.title(title_str)
        
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('平均训练损失 (Averaged Train Loss)')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        plt.savefig(full_output_path)
        print(f"损失图表已成功生成并保存为: '{full_output_path}'")
        plt.close()

    def draw_acc_and_place_in_folder(self):
        """
        主处理流程：扫描所有符合条件的日志文件，为它们生成图表并归档。
        """
        print("--- 开始扫描并处理所有以 'SLCDACP' 开头的日志文件和图片 ---")
        if not os.path.exists(self.root_path):
            print(f"错误：日志根目录 '{self.root_path}' 不存在。")
            return

        for filename in os.listdir(self.root_path):
            if filename.endswith(".log") and filename.startswith("SLCDACP"):
                try:
                    print(f"\n>>> 正在处理文件: {filename}")
                    drawer = draw_slcdacp(filename)
                    
                    if not drawer.output_dir:
                        print(f"跳过文件 '{filename}'，因为无法为其确定输出目录。")
                        continue

                    drawer.plot_acc_img()
                    drawer.plot_loss_img()

                    new_folder_path = drawer.output_dir 
                    
                    log_file_path = os.path.join(self.root_path, filename)
                    new_log_path = os.path.join(new_folder_path, filename)

                    if os.path.exists(log_file_path):
                        shutil.move(log_file_path, new_log_path)
                        print(f"已将日志文件移动到: '{new_log_path}'")
                
                except Exception as e:
                    print(f"处理文件 '{filename}' 时出错：{e}")
                    print("此文件的处理将跳过。")
        
        print("\n--- 所有符合条件的文件和图表已处理并整理完毕 ---")
        
    def flatten_logs_and_cleanup_images(self):
        """
        预处理函数：清理工作目录，将所有日志文件移动到根目录，为本次处理做准备。
        """
        print("--- 开始执行预处理任务：平整化日志目录并清理 ---")
        if not os.path.exists(self.root_path):
            print(f"错误：日志根目录 '{self.root_path}' 不存在。")
            return
        
        print(f"步骤 1: 正在扫描 '{self.root_path}' 及其子目录以移动 .log 文件...")
        for root, dirs, files in os.walk(self.root_path):
            if root == self.root_path:
                continue
            
            for file in files:
                if file.endswith(".log"):
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(self.root_path, file)
                    
                    if os.path.exists(destination_path):
                        print(f"警告：目标位置已存在文件 '{file}'，跳过移动。")
                        continue
                    
                    try:
                        shutil.move(source_path, destination_path)
                        print(f"已将 '{source_path}' 移动到根目录")
                    except Exception as e:
                        print(f"移动文件 '{source_path}' 时出错: {e}")

        print(f"\n步骤 2: 正在扫描当前工作目录 '{current_dir}' 以删除 .jpg 文件...")
        for filename in os.listdir(current_dir):
            if filename.lower().endswith(".jpg"):
                file_path = os.path.join(current_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"已删除图片: '{file_path}'")
                except Exception as e:
                    print(f"删除文件 '{file_path}' 时出错: {e}")

        print(f"\n步骤 3: 正在清理 '{self.root_path}' 目录下的所有子文件夹...")
        for filename in os.listdir(self.root_path):
            file_path = os.path.join(self.root_path, filename)
            if os.path.isdir(file_path):
                try:
                    shutil.rmtree(file_path)
                    print(f"已删除文件夹及其所有内容: '{file_path}'")
                except Exception as e:
                    print(f"删除文件夹 '{file_path}' 时出错: {e}")

        print("\n--- 预处理任务完成 ---")

    def _load_parameters_from_logger(self):
        """
        从日志文件的内容中通过正则表达式解析并加载参数。
        """
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return

        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            params = {
                'alpha': r'alpha\s*=\s*(\d+\.?\d*)',
                'batch_size': r'batch_size\s*=\s*(\d+)',
                'local_learning_rate': r'local_learning_rate\s*=\s*(\d+\.?\d*)',
                'purning_min': r'purning_min\s*=\s*(\d+\.?\d*)',
                'purning_base': r'purning_base\s*=\s*(\d+\.?\d*)',
                'purning_max': r'purning_max\s*=\s*(\d+\.?\d*)',
                'split_model_cnt': r'split_model_cnt\s*=\s*(\d+)',
                'prune_tool': r'prune_tool\s*=\s*([\w-]+)',
                'fixed_alpha': r'fixed_alpha\s*=\s*([\w./-]+)',
            }
            
            for key, pattern in params.items():
                match = re.search(pattern, content)
                if match:
                    value_str = match.group(1)
                    if key == 'fixed_alpha':
                        setattr(self, key, value_str)
                    elif key in ['alpha', 'purning_min', 'purning_base', 'purning_max', 'local_learning_rate']:
                        setattr(self, key, float(value_str))
                    elif key in ['batch_size', 'split_model_cnt']:
                        setattr(self, key, int(value_str))
                    else:
                        setattr(self, key, value_str)

        except Exception as e:
            print(f"解析日志文件 '{self.log_file_path}' 时发生错误：{e}")

if __name__ == '__main__':
    processor = draw_slcdacp()
    
    processor.flatten_logs_and_cleanup_images()
    
    processor.draw_acc_and_place_in_folder()