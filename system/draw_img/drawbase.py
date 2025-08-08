import torch
import os
import sys
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，防止在服务器上报错


class drawbase():
    def __init__(self,log_filename):

        self.root_path='system/logger'
        self.base_name = os.path.splitext(log_filename)[0]
        try:
            parts = self.base_name.rsplit('_', 4)
            self.algorithm, self.model_str, self.dataset, self.date_part, self.time_part = parts


            current_date = f"{self.date_part}_{self.time_part}"
        except ValueError:
            print(f"错误：文件名 '{log_filename}' 不符合 '{algorithm}_{model_str}_{dataset}_{date}_{time}.log' 的格式。")
            sys.exit(1) # 退出脚本



        
