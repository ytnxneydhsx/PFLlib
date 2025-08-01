import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

class dataselect():
    def __init__(self,mata_data):
        self.mata_data=mata_data
        self.data_dict=defaultdict(list)
        self.mata_data_sort()



    def mata_data_sort(self):
        for idx, (image_tensor, label_tensor)  in enumerate(self.mata_data):
            label_key = label_tensor.item()
            triplet = (idx, image_tensor, label_tensor)
            self.data_dict[label_key].append(triplet)
    





