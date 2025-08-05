# py3_main.py
import sys
import os
current_dir = os.getcwd()

print(current_dir)
sys.path.append(current_dir)
dataset_dir = os.path.join(current_dir, 'dataset')
sys.path.append(dataset_dir)
import configparser
from dataset.generate_MNIST import run_data_MNIST_generation
# 假设 py2_module 也已准备好
# from py2_module import print_all_configurations 


# def print_config_contents(config_object):
#     """
#     接收一个ConfigParser对象，并将其内容格式化打印出来。
#     """
#     print("\n--- [Config] 开始打印已加载的配置文件内容 ---")
#     # 检查config对象是否为空（比如文件没找到时）
#     if not config_object.sections():
#         print("[Config] 配置文件内容为空或未找到文件。")
#         return

#     # 遍历每一个配置段 [SECTION]
#     for section in config_object.sections():
#         print(f"[{section}]")
#         # 遍历该段下的每一个键值对
#         for key, value in config_object.items(section):
#             print(f"  {key} = {value}")
#     print("--- [Config] 内容打印完毕 ---\n")


if __name__ == "__main__":

    config = configparser.ConfigParser()
    files_read = config.read('system/config.ini')
    run_data_MNIST_generation(config)



