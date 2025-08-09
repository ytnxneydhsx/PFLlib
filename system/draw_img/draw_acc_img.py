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

class draw_acc_img(drawbase):
    def __init__(self, log_filename=None):
        super().__init__(log_filename)
        self.log_filename = log_filename
        
        # 仅当成功解析到属性时才构建目录
        if self.base_name and self.algorithm and self.model_str and self.dataset:
            alpha_str = f"alpha_{self.alpha}" if self.alpha is not None else "alpha_N-A"
            pruning_rate_str = f"data_pruning_rate_{self.data_pruning_rate}" if self.data_pruning_rate is not None else "pruning_rate_N-A"
            select_round_str = f"data_select_round_{self.data_select_round}" if self.data_select_round is not None else "select_round_N-A"
            time_str = f"time_{self.time}" if self.time is not None else "time_N-A"

            self.output_dir = os.path.join(
                self.root_path,
                self.algorithm,
                self.model_str,
                self.dataset,
                alpha_str,
                pruning_rate_str,
                select_round_str,
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
        if self.data_pruning_rate is not None:
            title_str += f"_pruning_rate:{self.data_pruning_rate}"
        if self.data_select_round is not None:
            title_str += f"_select_round:{self.data_select_round}"
        plt.title(title_str)
        
        plt.xlabel('Epoch')
        plt.ylabel('Averaged Test Accuracy')
        plt.grid(True)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        plt.savefig(full_output_path)
        print(f"图表已成功生成并保存为: '{full_output_path}'")
        plt.close()

    def draw_all_log_acc(self):
        """
        扫描 'system/logger' 目录下的所有 .log 文件，
        并为每个文件调用 plot_acc_img 方法生成图表。
        """
        print("--- 开始扫描并生成所有日志文件的准确率图表 ---")
        if not os.path.exists(self.root_path):
            print(f"错误：日志根目录 '{self.root_path}' 不存在。")
            return
        
        for filename in os.listdir(self.root_path):
            if filename.endswith(".log"):
                try:
                    print(f"\n正在处理文件: {filename}")
                    drawer = draw_acc_img(filename)
                    drawer.plot_acc_img()
                except Exception as e:
                    print(f"处理文件 '{filename}' 时出错：{e}")

        print("\n--- 所有日志文件的图表生成完毕 ---")

    def draw_acc_and_place_in_folder(self):
        """
        扫描 'system/logger' 目录下的所有 .log 文件，
        为每个文件生成图片，并将其和日志文件一起移动到以日志文件名为名的新文件夹中。
        """
        print("--- 开始扫描并处理所有日志文件和图片 ---")
        if not os.path.exists(self.root_path):
            print(f"错误：日志根目录 '{self.root_path}' 不存在。")
            return

        for filename in os.listdir(self.root_path):
            if filename.endswith(".log"):
                try:
                    base_name_without_ext = os.path.splitext(filename)[0]
                    log_file_path = os.path.join(self.root_path, filename)
                    
                    # 创建一个新实例来生成图片
                    drawer = draw_acc_img(filename)
                    drawer.plot_acc_img()

                    # 定义新文件夹的路径
                    new_folder_path = os.path.join(drawer.output_dir, base_name_without_ext)
                    os.makedirs(new_folder_path, exist_ok=True)
                    print(f"已为 '{filename}' 创建新文件夹: '{new_folder_path}'")

                    # 定义图片文件和日志文件的新路径
                    output_image_name = f"{base_name_without_ext}.jpg"
                    original_image_path = os.path.join(drawer.output_dir, output_image_name)
                    new_image_path = os.path.join(new_folder_path, output_image_name)
                    new_log_path = os.path.join(new_folder_path, filename)

                    # 移动图片文件
                    if os.path.exists(original_image_path):
                        shutil.move(original_image_path, new_image_path)
                        print(f"已将图片移动到: '{new_image_path}'")
                    
                    # 移动日志文件
                    if os.path.exists(log_file_path):
                        shutil.move(log_file_path, new_log_path)
                        print(f"已将日志文件移动到: '{new_log_path}'")
                        
                except Exception as e:
                    print(f"处理文件 '{filename}' 时出错：{e}")
                    print("此文件的处理将跳过。")
        
        print("\n--- 所有文件和图表已处理并整理完毕 ---")

# --- 调用示例 ---
if __name__ == '__main__':
    # 示例：仅生成所有图表，并按照原有目录结构存放
    # batch_drawer = draw_acc_img()
    # batch_drawer.draw_all_log_acc()
    
    # 示例：生成图表，并将图表和日志文件一起移动到新文件夹
    processor = draw_acc_img()
    processor.draw_acc_and_place_in_folder()