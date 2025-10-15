import re
import matplotlib.pyplot as plt
import os

def plot_logs(log_files, custom_labels):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False 
    fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    
    # 使用 zip 同时遍历文件路径和自定义标签
    for log_file, label in zip(log_files, custom_labels):
        accuracies = []
        losses = []
        rounds = []
        with open(log_file, 'r', encoding='utf-8') as f:
            round_count = 0
            for line in f:
                acc_match = re.search(r"Averaged Test Accuracy: (\d+\.\d+)", line)
                if acc_match:
                    accuracies.append(float(acc_match.group(1)))
                    rounds.append(round_count)
                    round_count += 1
                loss_match = re.search(r"Averaged Train Loss: ([\d.]+)", line)
                if loss_match:
                    losses.append(float(loss_match.group(1)))
        
        # 将标签设置为自定义名称
        ax_acc.plot(rounds, accuracies, marker='None', linestyle='-', label=label, linewidth=2.5)
        num_points = min(len(rounds), len(losses))
        ax_loss.plot(rounds[:num_points], losses[:num_points], marker='x', linestyle='--', label=label)
    ax_acc.tick_params(axis='both', which='major', labelsize=20)
    ax_loss.tick_params(axis='both', which='major', labelsize=15)
    ax_acc.set_xlabel("Training Round", fontsize=20)
    ax_acc.set_ylabel("Test Accuracy(%)", fontsize=20)
    # 将图例放在右上角
    ax_acc.legend(fontsize=20, loc='upper right')
    ax_acc.grid(True)
    ax_loss.set_title("模型损失 (Loss) 变化趋势", fontsize=20)
    ax_loss.set_xlabel("训练轮次 (Round)", fontsize=12)
    ax_loss.set_ylabel("平均训练损失 (Averaged Train Loss)", fontsize=12)
    # 将图例放在右上角
    ax_loss.legend(fontsize=12, loc='upper right')
    ax_loss.grid(True)
    fig_acc.savefig("accuracy_plot.png")
    fig_loss.savefig("loss_plot.png")

if __name__ == '__main__':
    log_file_paths = [
        '/mnt/tjl/PFLlib/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/fixed_alpha_mask_grad/purning_min_0.9/purning_base_0.9/purning_max_0.9/batch_size_128/lr_0.0015/optimizer_SGD/global_rounds_200.0/split_model_cnt_4/time_2025-10-13_17-31-03/1b5f74d33ddc82634741eb9b12019b0cfc5e7b2d.log',
        '/mnt/tjl/PFLlib/system/logger/SLCDACP/VGG16/Cifar10/alpha_1.0/fixed_alpha_mask_grad/purning_min_0.9/purning_base_0.9/purning_max_0.9/batch_size_128/lr_0.0015/optimizer_SGD/global_rounds_200.0/split_model_cnt_4/time_2025-10-13_17-31-08/7f5b72be84882a330ab2c7a663540ee24f7c1870.log'


    ]
    
    # 在这里定义你想要的自定义名称，顺序要和文件列表对应
    custom_labels = ['default','qu']

    # 将自定义名称列表传递给函数
    plot_logs(log_file_paths, custom_labels)
    

