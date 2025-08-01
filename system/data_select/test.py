import numpy as np


# 假设 data.npz 文件存在，并且包含 'array1' 和 'array2'
# 这里我们先创建一个示例 npz 文件用于演示
array1 = np.array([1, 2, 3])
array2 = np.array([[4, 5], [6, 7]])
np.savez('data.npz', array1=array1, array2=array2)

# 读取 npz 文件
with np.load('data.npz') as data:
    # 查看文件中包含的所有数组的名称
    print("文件中的数组名称:", data.files)

    # 访问特定的数组
    arr1 = data['array1']
    arr2 = data['array2']

    print("Array 1:", arr1)
    print("Array 2:", arr2)

    # 如果你知道所有数组的名称，也可以这样遍历：
    for key in data.files:
        print(f"数组 '{key}':\n{data[key]}")