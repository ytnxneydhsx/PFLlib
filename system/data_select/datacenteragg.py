import numpy as np
import torchvision.transforms as transforms
from data_select.dataselectbase import dataselect
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
import os 
import torch
import matplotlib
matplotlib.use('Agg')


class datacenteragg(dataselect):
    def __init__(self, mata_data,Pruning_rate=0.8):
        super().__init__(mata_data)
        self.clustered_data_sum = defaultdict(dict)
        self.clustered_original_center_sum = defaultdict(list)
        self.data_distance_value_sum = defaultdict(dict)
        self.center_coordinates=defaultdict(dict)
        self.Pruning_mata_data=[]
        self.global_center_coordinates=None
        self.Pruning_rate=Pruning_rate
        self.select_data_clusters()


        # self.get_center_coordinates()


    def select_data_clusters(self):
        for key, triplets in self.data_dict.items():
            if not triplets:
                continue
            center_tensor = torch.mean(torch.stack([t[1] for t in triplets]), dim=0)
            self.clustered_original_center_sum[key] = [center_tensor]
            self.clustered_data_sum[key] = {0: triplets}
        self.center_coordinates=self.clustered_original_center_sum


    def get_data_distance_value(self):
        all_global_centers = list(self.global_center_coordinates.values())
        self.data_distance_value_sum = {}
        for key, clusters in self.clustered_data_sum.items():
            for cluster_id, data_triplets in clusters.items(): 
                results_for_cluster = []
                for original_tuple in data_triplets:
                    index, tensor_data, label = original_tuple
                    distances = [torch.norm(tensor_data - center).item() for center in all_global_centers]
                    distances.sort()
                    closest_dist = distances[0]
                    second_closest_dist = distances[1]
                    final_value = second_closest_dist - closest_dist
                    results_for_cluster.append((index, final_value, label))
                if key not in self.data_distance_value_sum:
                    self.data_distance_value_sum[key] = {}
                self.data_distance_value_sum[key][cluster_id] = results_for_cluster

    def clusters_data_sort(self):
        for key, clusters in self.data_distance_value_sum.items():
            for cluster_id, data_list in clusters.items():
                data_list.sort(key=lambda x: x[1], reverse=False)

    def Pruning_data(self,train_data):
        for key,clusters in self.data_distance_value_sum.items():
            for cluster_id, data_triplets in clusters.items(): 
                triplets_len=len(data_triplets)
                Pruning_mata_data_len=int(triplets_len*self.Pruning_rate)
                cnt=0
                for original_tuple in data_triplets:
                    if cnt==Pruning_mata_data_len:
                        break
                    else:
                        index, tensor_data, label = original_tuple
                        original_train_data=train_data[index][0]
                        self.Pruning_mata_data.append((original_train_data,label))
                        cnt=cnt+1
        return self.Pruning_mata_data
                    

    def Per_Pruning_data(self):
        self.get_data_distance_value()
        self.clusters_data_sort()















    



    # def get_data_distance_value(self):
    #     for key in self.center_coordinates.keys():
    #         self.data_distance_value_sum[key] = {}
    #         for triplet in self.data_dict[key]:
    #             index, tensor_data, label = triplet
    #             distance = torch.norm(tensor_data - self.center_coordinates[key])
    #             self.data_distance_value_sum[key][index] = distance.item()






