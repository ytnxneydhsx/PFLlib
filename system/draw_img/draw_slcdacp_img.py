import os
import re
import matplotlib.pyplot as plt
import matplotlib
from drawbase import drawbase
import sys
import shutil

# Set matplotlib backend to prevent errors in a GUI-less environment
matplotlib.use('Agg')

class draw_slcdacp(drawbase):
    def __init__(self, log_filename=None):
        super().__init__(log_filename)
        self.log_filename = log_filename
        # Initialize attributes
        self.algorithm = None
        self.model_str = None
        self.dataset = None
        self.current_date = None
        self.local_learning_rate = None
        self.purning_min = None
        self.purning_base = None
        self.purning_max = None
        self.prune_tool = None
        self.split_model_cnt = None
        self.fixed_alpha = None
        self.batch_size = None
        self.global_rounds = None
        self.optimizer_str = None
        self.momentum= None
        self._load_parameters_from_logger()

        # Construct the base_name using the extracted parameters
        if self.algorithm and self.model_str and self.dataset and self.current_date:
            self.base_name = f"{self.algorithm}_{self.model_str}_{self.dataset}_{self.current_date}"
        if self.base_name and self.algorithm and self.model_str and self.dataset:
            alpha_str = f"alpha_{self.alpha}" if self.alpha is not None else "alpha_N-A"
            lr_str = f"lr_{self.local_learning_rate}" if self.local_learning_rate is not None else "lr_N-A"
            time_str = f"time_{self.current_date}" if self.current_date is not None else "time_N-A"
            purning_min_str = f"purning_min_{self.purning_min}" if self.purning_min is not None else "purning_min_N-A"
            purning_base_str = f"purning_base_{self.purning_base}" if self.purning_base is not None else "purning_base_N-A"
            purning_max_str = f"purning_max_{self.purning_max}" if self.purning_max is not None else "purning_max_N-A"
            prune_tool_str = f"{self.prune_tool}" if self.prune_tool is not None else "default"
            batch_size_str = f"batch_size_{self.batch_size}" if self.batch_size is not None else "batch_size_N-A"
            optimizer_str = f"optimizer_{self.optimizer_str}" if self.optimizer_str is not None else "SGD"
            global_rounds_str = f"global_rounds_{self.global_rounds}" if self.global_rounds is not None else "global_rounds_N-A"
            split_model_cnt_str = f"split_model_cnt_{self.split_model_cnt}" if self.split_model_cnt is not None else "split_model_cnt_N-A"
            path_components = [
                self.root_path, self.algorithm, self.model_str, self.dataset,
                alpha_str, prune_tool_str, purning_min_str, purning_base_str,
                purning_max_str, batch_size_str, lr_str, optimizer_str, global_rounds_str,
                split_model_cnt_str,
            ]
            if prune_tool_str == 'fixed_alpha' and self.fixed_alpha is not None:
                path_components.extend(str(self.fixed_alpha).split('/'))
            path_components.append(time_str)
            self.output_dir = os.path.join(*path_components)
            self.output_dir = self.output_dir.replace(':', '_')
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory is ready: '{self.output_dir}'")
        else:
            self.output_dir = None
            print("Warning: Could not create output directory.")

    def plot_acc_img(self):
        if not self.log_filename or not self.output_dir: return
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        accuracies = []
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Averaged Test Accuracy:" in line:
                        match = re.search(r"(\d+\.\d+)", line)
                        if match: accuracies.append(float(match.group(1)))
        except FileNotFoundError: return
        plt.figure(figsize=(12, 7)); plt.plot(accuracies, linestyle='-', color='b')
        plt.title(f"{self.base_name} - Accuracy"); plt.xlabel('Epoch'); plt.ylabel('Averaged Test Accuracy')
        plt.grid(True); plt.tight_layout(); plt.savefig(full_output_path)
        print(f"Accuracy chart saved: '{full_output_path}'")
        plt.close()

    def plot_loss_img(self):
        if not self.log_filename or not self.output_dir: return
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}_loss.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        losses = []
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Averaged Train Loss:" in line:
                        match = re.search(r"(\d+\.\d+)", line)
                        if match: losses.append(float(match.group(1)))
        except FileNotFoundError: return
        plt.figure(figsize=(12, 7)); plt.plot(losses, linestyle='--', color='r')
        plt.title(f"{self.base_name} - Loss"); plt.xlabel('Epoch'); plt.ylabel('Averaged Train Loss')
        plt.grid(True); plt.tight_layout(); plt.savefig(full_output_path)
        print(f"Loss chart saved: '{full_output_path}'")
        plt.close()

    def plot_pruning_rate_img(self):
        if not self.log_filename or not self.output_dir: return
        full_log_path = os.path.join(self.root_path, self.log_filename)
        output_image_name = f"{self.base_name}_pruning_rate.jpg"
        full_output_path = os.path.join(self.output_dir, output_image_name)
        per_round_rates = []
        pattern = re.compile(r"Overall Average Pruning Rate for Round \d+: (\d+\.\d+)")
        try:
            with open(full_log_path, 'r', encoding='utf-8') as f:
                per_round_rates = [float(rate) for rate in pattern.findall(f.read())]
        except FileNotFoundError: return
        if not per_round_rates: return
        cumulative = [sum(per_round_rates[:i+1]) / (i + 1) for i in range(len(per_round_rates))]
        global_avg = sum(per_round_rates) / len(per_round_rates)
        global_line = [global_avg] * len(per_round_rates)
        plt.figure(figsize=(12, 7))
        plt.plot(per_round_rates, linestyle='-', color='g', label='Current Round Avg')
        plt.plot(cumulative, linestyle='--', color='r', label='Cumulative Avg')
        plt.plot(global_line, linestyle=':', color='b', label=f'Global Avg: {global_avg:.4f}')
        plt.title(f"{self.base_name} - Pruning Rates"); plt.xlabel('Epoch'); plt.ylabel('Average Pruning Rate')
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(full_output_path)
        print(f"Pruning rate chart saved: '{full_output_path}'")
        plt.close()

    def draw_acc_and_place_in_folder(self):
        print("--- Starting to scan and process log files ---")
        if not os.path.exists(self.root_path): return
        for filename in os.listdir(self.root_path):
            # if filename.endswith(".log") and filename.startswith("SLCDACP"):
            if filename.endswith(".log"):
                try:
                    print(f"\n>>> Processing file: {filename}")
                    drawer = draw_slcdacp(filename)
                    if not drawer.output_dir:
                        print(f"Skipping file '{filename}', cannot determine output directory.")
                        continue
                    drawer.plot_acc_img()
                    drawer.plot_loss_img()
                    drawer.plot_pruning_rate_img()
                    log_path = os.path.join(self.root_path, filename)
                    new_log_path = os.path.join(drawer.output_dir, filename)
                    if os.path.exists(log_path):
                        shutil.move(log_path, new_log_path)
                        print(f"Log file moved to: '{new_log_path}'")
                except Exception as e:
                    print(f"Error processing '{filename}': {e}")
        print("\n--- All files processed ---")
    
    def flatten_logs_and_cleanup_images(self):
        print("--- Starting preprocessing tasks ---")
        if not os.path.exists(self.root_path): return
        for root, _, files in os.walk(self.root_path):
            if root == self.root_path: continue
            for file in files:
                if file.endswith(".log"):
                    try: shutil.move(os.path.join(root, file), os.path.join(self.root_path, file))
                    except Exception as e: print(f"Error moving '{file}': {e}")
        for filename in os.listdir(self.root_path):
            file_path = os.path.join(self.root_path, filename)
            if filename.lower().endswith(".jpg"):
                try: os.remove(file_path)
                except Exception as e: print(f"Error deleting '{file_path}': {e}")
            elif os.path.isdir(file_path):
                try: shutil.rmtree(file_path)
                except Exception as e: print(f"Error deleting folder '{file_path}': {e}")
        print("--- Preprocessing complete ---")

    def _load_parameters_from_logger(self):
        if not self.log_file_path or not os.path.exists(self.log_file_path): return
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f: content = f.read()
            params = {
                # Added parameters
                'algorithm': r'algorithm\s*=\s*(\w+)',
                'model_str': r'model\s*=\s*(\w+)',
                'dataset': r'dataset\s*=\s*(\w+)',
                'current_date': r'current_date\s*=\s*(\S+)',
                'alpha': r'alpha\s*=\s*(\d+\.?\d*)', 
                'batch_size': r'batch_size\s*=\s*(\d+)',
                'local_learning_rate': r'local_learning_rate\s*=\s*(\d+\.?\d*)',
                'purning_min': r'purning_min\s*=\s*(\d+\.?\d*)', 
                'purning_base': r'purning_base\s*=\s*(\d+\.?\d*)',
                'purning_max': r'purning_max\s*=\s*(\d+\.?\d*)', 
                'split_model_cnt': r'split_model_cnt\s*=\s*(\d+)',
                'prune_tool': r'prune_tool\s*=\s*([\w-]+)', 
                'fixed_alpha': r'fixed_alpha\s*=\s*([\w./-]+)',
                'global_rounds': r'global_rounds\s*=\s*(\d+\.?\d*)',
                'optimizer_str': r'optimizer_str\s*=\s*(\w+)',
            }
            for key, pattern in params.items():
                match = re.search(pattern, content)
                if match:
                    val = match.group(1).strip()
                    # Assign as string or number based on the key
                    if key in ['fixed_alpha', 'prune_tool', 'optimizer_str', 'algorithm', 'model_str', 'dataset', 'current_date']: 
                        setattr(self, key, val)
                    elif key in ['alpha', 'local_learning_rate', 'purning_min', 'purning_base', 'purning_max', 'global_rounds']: 
                        setattr(self, key, float(val))
                    else: 
                        setattr(self, key, int(val))
        except Exception as e: print(f"Error parsing log file: {e}")

if __name__ == '__main__':
    processor = draw_slcdacp()
    # To run cleanup, uncomment the following line
    # processor.flatten_logs_and_cleanup_images() 
    processor.draw_acc_and_place_in_folder()