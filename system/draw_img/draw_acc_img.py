import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，防止在服务器上报错
from drawbase import drawbase
import sys
current_dir = os.getcwd()
print(current_dir)
sys.path.append(current_dir)
class draw_acc_img(drawbase):
    def __init__(self,log_filename):
        super().__init__(log_filename)
        self.log_filename=log_filename
        self.output_dir = os.path.join(self.root_path, self.algorithm, self.model_str, self.dataset)
        os.makedirs(self.output_dir, exist_ok=True) # exist_ok=True表示如果文件夹已存在则不报错
        print(f"输出目录已准备好: '{self.output_dir}'")
    def plot_acc_img(self):
        full_log_path = os.path.join(self.root_path,self.log_filename)
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
        plt.title('Averaged Test Accuracy Over Training Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Averaged Test Accuracy')
        plt.grid(True)
        plt.tight_layout()
        # --- 将图表保存到指定的输出路径 ---
        # 确保输出目录存在
        output_dir = os.path.dirname(full_output_path)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(full_output_path)
        print(f"图表已成功生成并保存为: '{full_output_path}'")

a=draw_acc_img('SLCS_VGG16_Cifar10_2025-08-07_19-31-11.log')
a.plot_acc_img()