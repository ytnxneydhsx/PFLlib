import os
import sys
import re
import matplotlib.pyplot as plt
import matplotlib
import configparser

matplotlib.use('Agg')  # 使用非交互式后端，防止在服务器上报错
current_dir = os.getcwd()
sys.path.append(current_dir)

class drawbase():
    def __init__(self, log_filename=None): # 将 log_filename 设置为可选参数
        self.root_path = 'system/logger'
        self.log_filename = log_filename
        self.base_name = None
        self.log_file_path = None

        # 定义需要从 logger 中提取的属性

        self.alpha = None
        self.batch_size = None
        self.time = None
        self.algorithm = None
        self.model_str = None
        self.dataset = None

        # 仅当传入 log_filename 时才进行解析
        if log_filename:
            self.base_name = os.path.splitext(log_filename)[0]
            self.log_file_path = os.path.join(self.root_path, log_filename)

            try:
                # 匹配文件名格式: SLCS_VGG16_Cifar10_2025-08-09_00-42-22
                match = re.search(r'(.+?)_(.+?)_(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', self.base_name)
                
                if match:
                    self.algorithm, self.model_str, self.dataset, self.time = match.groups()
                else:
                    # 如果文件名不匹配，则设置默认值
                    self.algorithm = "Unknown"
                    self.model_str = "Unknown"
                    self.dataset = "Unknown"
                    print(f"警告：文件名 '{log_filename}' 格式不完全匹配，部分属性将为默认值。")
            except ValueError:
                print(f"错误：文件名 '{log_filename}' 不符合预期格式。")
                # 不退出，让子类处理文件不存在的情况
                pass
            