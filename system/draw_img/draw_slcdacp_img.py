import os
import re
import matplotlib.pyplot as plt
import matplotlib
from drawbase import drawbase
import sys
import shutil

matplotlib.use('Agg')
current_dir = os.getcwd()
sys.path.append(current_dir)
class draw_slcdacp(drawbase):
    def __init__(self, log_filename=None):
        super().__init__(log_filename)
        self.log_filename = log_filename
        self.purning_min=None
        self.purning_base=None
        self.purning_max=None
        self._load_parameters_from_logger()


        # 仅当成功解析到属性时才构建目录
        if self.base_name and self.algorithm and self.model_str and self.dataset:
            alpha_str = f"alpha_{self.alpha}" if self.alpha is not None else "alpha_N-A"
            time_str = f"time_{self.time}" if self.time is not None else "time_N-A"
            purning_min_str = f"purning_min_{self.purning_min}" if self.purning_min is not None else "purning_min_N-A"
            purning_base_str = f"purning_base_{self.purning_base}" if self.purning_base is not None else "purning_base_N-A"
            purning_max_str = f"purning_max_{self.purning_max}" if self.purning_max is not None else "purning_max_N-A"



            self.output_dir = os.path.join(
                self.root_path,
                self.algorithm,
                self.model_str,
                self.dataset,
                alpha_str,
                purning_min_str,
                purning_base_str,
                purning_max_str,
                time_str
            )
            # 由于 time 字符串中可能包含冒号，需要额外处理
            # 这里的 time_str 格式为 'time:2025-08-09_12-16-52'，需要把 time 后的冒号和时间中的冒号都处理掉
            # 将 'time:2025-08-09_12-16-52' 变为 'time_2025-08-09_12-16-52'
            # 还需要将时间中的冒号 '-' 替换为下划线
            self.output_dir = self.output_dir.replace(':', '_')
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"输出目录已准备好: '{self.output_dir}'")
        else:
            self.output_dir = None
            print(f"警告：由于文件名未提供或格式不正确，无法创建输出目录。")

    def plot_acc_img(self):
        if not self.log_filename or not self.output_dir:
            print("错误：未指定日志文件或输出目录，无法绘制图表。")
            return
        
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        print(f"正在从 '{full_log_path}' 读取数据并生成图表...")
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
            title_str += f"purning_min:{self.purning_min}"
        if self.purning_base is not None:
            title_str += f"purning_base:{self.purning_base}"
        if self.purning_max is not None:
            title_str += f"purning_max:{self.purning_max}"
        plt.title(title_str)
        
        plt.xlabel('Epoch')
        plt.ylabel('Averaged Test Accuracy')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        plt.savefig(full_output_path)
        print(f"图表已成功生成并保存为: '{full_output_path}'")
        plt.close()


    def draw_acc_and_place_in_folder(self):

        print("--- 开始扫描并处理所有以 'SLCDACP' 开头的日志文件和图片 ---")
        if not os.path.exists(self.root_path):
            print(f"错误：日志根目录 '{self.root_path}' 不存在。")
            return

        for filename in os.listdir(self.root_path):
            # --- 关键改动在这里 ---
            # 增加一个条件，确保只处理以 'SLCS' 开头的 .log 文件
            if filename.endswith(".log") and filename.startswith("SLCDACP"):
                try:
                    print(f"\n>>> 正在处理文件: {filename}")
                    # 创建一个新实例来生成图片和目录结构
                    drawer = draw_slcdacp(filename)
                    
                    # 检查drawer是否成功创建了output_dir
                    if not drawer.output_dir:
                        print(f"跳过文件 '{filename}'，因为无法为其确定输出目录。")
                        continue

                    drawer.plot_acc_img()

                    # 定义新文件夹的路径 (注意：这里我们直接使用drawer.output_dir作为目标)
                    # output_dir 已经是根据文件属性创建的结构化路径
                    new_folder_path = drawer.output_dir 
                    
                    # 定义图片文件和日志文件的原始路径
                    base_name_without_ext = os.path.splitext(filename)[0]
                    log_file_path = os.path.join(self.root_path, filename)
                    output_image_name = f"{base_name_without_ext}.jpg"
                    # 图片生成后会直接位于 output_dir 中
                    original_image_path = os.path.join(drawer.output_dir, output_image_name)

                    # 定义日志文件的新路径
                    new_log_path = os.path.join(new_folder_path, filename)

                    # 移动日志文件 (图片已经生成在目标位置，无需移动)
                    if os.path.exists(log_file_path):
                        shutil.move(log_file_path, new_log_path)
                        print(f"已将日志文件移动到: '{new_log_path}'")
                    
                    # 简单清理一下逻辑，因为图片已经直接生成在目标目录，所以不需要移动图片
                    # 只需要移动日志文件即可

                except Exception as e:
                    print(f"处理文件 '{filename}' 时出错：{e}")
                    print("此文件的处理将跳过。")
            # --- 改动结束 ---
        
        print("\n--- 所有符合条件的文件和图表已处理并整理完毕 ---")

    def _load_parameters_from_logger(self):
        """
        从日志文件的内容中解析并加载参数。
        """
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            print(f"警告：日志文件 '{self.log_file_path}' 不存在或未指定，跳过参数加载。")
            return

        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            params = {
                'alpha': r'alpha = (\d+\.\d+)',
                'batch_size': r'batch_size = (\d+)',
                'purning_min': r'purning_min = (\d+\.\d+)',
                'purning_base': r'purning_base = (\d+\.\d+)',
                'purning_max': r'purning_max = (\d+\.\d+)',
            }
            for key, pattern in params.items():
                                match = re.search(pattern, content)
                                if match:
                                    value_str = match.group(1)
                                    if key in ['data_pruning_rate', 'alpha', 'purning_min', 'purning_base', 'purning_max']:
                                        setattr(self, key, float(value_str))
                                    elif key in ['data_select_round', 'batch_size']:
                                        setattr(self, key, int(value_str))
                                    else:
                                        setattr(self, key, value_str)

        except Exception as e:
            print(f"解析日志文件 '{self.log_file_path}' 时发生错误：{e}")
            sys.exit(1)

if __name__ == '__main__':

    processor = draw_slcdacp()
    processor.draw_acc_and_place_in_folder()