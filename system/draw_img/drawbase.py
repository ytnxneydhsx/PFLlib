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
        self.data_pruning_rate = None
        self.data_select_round = None
        self.alpha = None
        self.batch_size = None
        self.hook_layer_name = None
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
            
            # 从日志文件中读取其他参数
            self._load_parameters_from_logger()

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
                'data_pruning_rate': r'data_pruning_rate = (\S+)',
                'data_select_round': r'data_select_round = (\d+)',
                'alpha': r'alpha = (\d+\.\d+)',
                'batch_size': r'batch_size = (\d+)',
                'hook_layer_name': r'hook_layer_name = (\S+)',
            }

            for key, pattern in params.items():
                match = re.search(pattern, content)
                if match:
                    value_str = match.group(1)
                    if key in ['data_pruning_rate', 'alpha']:
                        setattr(self, key, float(value_str))
                    elif key in ['data_select_round', 'batch_size']:
                        setattr(self, key, int(value_str))
                    else:
                        setattr(self, key, value_str)

        except Exception as e:
            print(f"解析日志文件 '{self.log_file_path}' 时发生错误：{e}")
            sys.exit(1)