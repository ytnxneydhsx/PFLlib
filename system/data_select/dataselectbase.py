import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

class dataselect():
    def __init__(self,mata_data):
        
        self.mata_data=mata_data
        self.data_dict=defaultdict(list)
        self.mata_data_to_dict()



    def mata_data_to_dict(self):
            # 在处理前清空，确保每次调用都是从零开始
            # self.data_dict.clear()
            print(f"--- dataselect: 开始处理类型为 {type(self.mata_data)} 的数据 ---")
            # --- 1. 判断 self.mata_data 是否为列表 ---
            if isinstance(self.mata_data, list):
                print("检测到数据类型为列表，按 'enumerate' 方式进行处理...")
                if not self.mata_data:
                    print("警告：输入列表为空。")
                    return

                # 对列表中的第一项进行结构验证，确保是 (张量, 张量) 的形式
                first_item = self.mata_data[0]
                if not (isinstance(first_item, (list, tuple)) and len(first_item) == 2):
                    raise TypeError("列表数据源的格式不正确！期望的格式为 [(数据, 标签), ...]")

                # 验证通过，开始处理
                for idx, (data_tensor, label_tensor) in enumerate(self.mata_data):
                    label_key = label_tensor.item()
                    triplet = (idx, data_tensor, label_tensor)
                    self.data_dict[label_key].append(triplet)

            # --- 2. 判断 self.mata_data 是否为字典 ---
            elif isinstance(self.mata_data, dict):
                print("检测到数据类型为字典，按 '.items()' 方式进行处理...")
                if not self.mata_data:
                    print("警告：输入字典为空。")
                    return
                # 对字典中的第一项进行结构验证，确保值的格式是 (张量, 张量)
                first_key = next(iter(self.mata_data))
                first_value = self.mata_data[first_key]
                if not (isinstance(first_value, (list, tuple)) and len(first_value) == 2):
                    raise TypeError("字典数据源的格式不正确！期望的格式为 {索引: (数据, 标签)}")

                # 验证通过，开始处理
                for idx, (data_tensor, label_tensor) in self.mata_data.items():
                    label_key = label_tensor.item()
                    triplet = (idx, data_tensor, label_tensor)
                    self.data_dict[label_key].append(triplet)
            # --- 3. 如果是未知类型，则报错 ---
            else:
                raise TypeError(f"不支持的数据源类型: {type(self.mata_data)}。dataselect 只支持列表或字典。")
            print(f"处理完成，数据已按 {len(self.data_dict)} 个标签分组存入 self.data_dict。")
    





