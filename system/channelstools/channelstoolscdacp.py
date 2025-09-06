import torch
import torch.nn as nn
from typing import List, Union
from typing import List, Tuple
from channelstools.channelstoolsbase import channelstoolsbase

class channelstoolcdacp(channelstoolsbase):
    def __init__(self, args):
        super().__init__(args)
        self.channel_group_num=args.channel_group_num


    def get_channels_freeze_list_sum(self):
        group_size = self.channel_num // self.channel_group_num
        channels_freeze_list_sum = []
        for i in range(self.channel_group_num):
            start_channel = i * group_size
            if i == self.channel_group_num - 1:
                end_channel = self.channel_num
            else:
                end_channel = (i + 1) * group_size
            channels_freeze_list = list(range(start_channel, end_channel))
            channels_freeze_list_sum.append(channels_freeze_list)
        return channels_freeze_list_sum
    
    def analyze_layer_with_data(self,channels_freeze_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)
        inputs, labels = self.bath_data
        inputs, labels = inputs.to(device), labels.to(device)
        sequential_layers = list(self.model.children())
        if not 1 <= self.split_model_cnt <= len(sequential_layers):
            raise IndexError(
                f"Layer index {self.split_model_cnt} is out of bounds. Model has {len(sequential_layers)} sequential modules."
            )
        target_layer = sequential_layers[self.split_model_cnt - 1]
        criterion = nn.CrossEntropyLoss()
        hook_handle = None
        if channels_freeze_list:
            def create_freezer_hook(channels):
                def hook(module, input, output):
                    if output.dim() == 4:  
                        output[:, channels, :, :] = 0.0
                    elif output.dim() == 2:  
                        output[:, channels] = 0.0
                    return output
                return hook
            hook = create_freezer_hook(channels_freeze_list)
            hook_handle = target_layer.register_forward_hook(hook)
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
        if hook_handle is not None:
            hook_handle.remove()
        return loss.item()
