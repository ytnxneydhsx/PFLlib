import numpy as np
import torchvision.transforms as transforms
from data_select.dataselectbase import dataselect
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
import os 



class datakmeans(dataselect):
    def __init__(self, mata_data,per_class=100):
        super().__init__(mata_data)
        self.per_class=per_class
        self.clustered_data_sum=defaultdict(dict)
        self.clustered_center_sum=defaultdict(list)
        self.select_data_clusters()
        self.data_distance_value_sum=defaultdict(dict)
        self.get_data_distance_value()
        self.clusters_data_sort()
        print(111111)


    def select_data_clusters(self):
        for key in self.data_dict.keys():
            label_len=len(self.data_dict[key])
            n_clusters = max(1, int(label_len / self.per_class)) # Limit to 10 clusters or fewer
            kmeans = KMeans(n_clusters, random_state=0, n_init=10)
            all_tensors_list = []
            for original_tuple in self.data_dict[key]:
                index, tensor_data, label = original_tuple
                all_tensors_list.append(tensor_data)
            X = np.array([t.numpy().flatten() for t in all_tensors_list])
            kmeans.fit(X)
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            clustered_data = defaultdict(list)
            for i, cluster_label in enumerate(cluster_labels):
                original_tuple = self.data_dict[key][i]
                clustered_data[cluster_label].append(original_tuple)
            clusetr_center= []
            for i,clusetr_center_value in enumerate(cluster_centers):
                clusetr_center.append(clusetr_center_value)

            self.clustered_data_sum[key] = clustered_data
            self.clustered_center_sum[key]=clusetr_center



    def clusters_data_sort(self):
        for key in self.clustered_data_sum.keys():
            for cluster in self.clustered_data_sum[key]:
                self.data_distance_value_sum[key][cluster].sort(key=lambda x: x[1], reverse=False)



    # 单个点的距离
    def get_data_distance_value(self):
        for key in self.clustered_data_sum.keys():
            data_distance_value=defaultdict(list)
            for cluster in self.clustered_data_sum[key]:
                for original_tuple in self.clustered_data_sum[key][cluster]:
                    index, tensor_data, label = original_tuple
                    center_value = self.clustered_center_sum[key][cluster]
                    distance = np.linalg.norm(tensor_data.numpy().flatten() - center_value)
                    data_distance_value[cluster].append((index, distance, label))
            self.data_distance_value_sum[key]=data_distance_value




   #最近的两个簇中心的距离差
    # def get_data_distance_value(self):
    #     for key in self.clustered_data_sum.keys():
    #         data_distance_value = defaultdict(list)
    #         # 获取当前原始标签下的所有簇中心
    #         current_label_all_centers = self.clustered_center_sum[key]
    #         # 遍历每个簇
    #         for cluster in self.clustered_data_sum[key]:
    #             # 遍历每个簇中的数据点
    #             for original_tuple in self.clustered_data_sum[key][cluster]:
    #                 index, tensor_data, label = original_tuple
    #                 # 展平数据点，用于距离计算
    #                 data_point = tensor_data.numpy().flatten()
    #                 # 计算当前数据点到所有簇中心的距离
    #                 distances = [np.linalg.norm(data_point - center) for center in current_label_all_centers]
    #                 # 对距离进行排序，找到最小的两个距离
    #                 distances.sort()
    #                 min_distance_1 = distances[0]
    #                 min_distance_2 = distances[1]
    #                 # 计算距离差
    #                 distance_diff = min_distance_2 - min_distance_1
    #                 # 将结果存储到字典中
    #                 data_distance_value[cluster].append((index, distance_diff, label))
    #         self.data_distance_value_sum[key] = data_distance_value
    #     return self.data_distance_value_sum






    def show_data_img(self):
        for key in self.data_distance_value_sum.keys():
            cluster_data = self.data_distance_value_sum[key]
            for cluster_id, sorted_distances in cluster_data.items():
                if len(sorted_distances) < 4:
                    print(f"警告: 标签 '{key}' 的簇 {cluster_id} 样本数不足4个，跳过绘图。")
                    continue
                
                max_dist_tuples = sorted_distances[-2:]
                min_dist_tuples = sorted_distances[:2]
                selected_tuples_with_diff = sorted(max_dist_tuples + min_dist_tuples, key=lambda x: x[1], reverse=True)
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
                plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像时负号 '-' 显示为方块
                fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                fig.suptitle(f"标签 '{key}' - 簇 {cluster_id}", fontsize=16)
                titles = [
                    f"Max Distance Diff: {selected_tuples_with_diff[0][1]:.4f}", 
                    f"Second Max Diff: {selected_tuples_with_diff[1][1]:.4f}", 
                    f"Min Distance Diff: {selected_tuples_with_diff[2][1]:.4f}", 
                    f"Second Min Diff: {selected_tuples_with_diff[3][1]:.4f}"
                ]
                for i, ax in enumerate(axes.flatten()):
                    original_index = selected_tuples_with_diff[i][0]
                    image_tensor = self.mata_data[original_index][0]
                    image_np = image_tensor.squeeze().numpy()
                    ax.imshow(image_np, cmap='gray')
                    ax.set_title(titles[i], fontsize=10)
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
                filename = 'my_image.png'
                plt.savefig(filename)
                full_path = os.path.abspath(filename)
                print(f"文件已成功保存到: {full_path}")



