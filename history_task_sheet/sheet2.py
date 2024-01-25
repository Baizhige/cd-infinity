import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import os
import torch
def generate_naive_T(m, n):
    if m <= n:
        raise ValueError("m must be greater than n")

    # 创建单位阵部分
    T = np.eye(n)

    # 生成额外的行
    extra_rows = []
    for positions in combinations(range(n), 2):
        if len(extra_rows) < m - n:
            new_row = np.zeros(n)
            i, j = positions
            new_row[i] = 0.5
            new_row[j] = 0.5
            extra_rows.append(new_row)
        else:
            break

    # 将额外的行添加到 T
    T = np.vstack((T, extra_rows))

    return T

def clean_T(T, n):
    # 初始化一个与 T 形状相同的零矩阵
    clean_matrix = np.zeros_like(T)

    # 对于 T 的每一行
    for i in range(T.shape[0]):
        # 找到前 n 个最大元素的索引
        top_n_idx = np.argsort(T[i, :])[-n:]

        # 将这些位置的值设为 1
        clean_matrix[i, top_n_idx] = 1

    return clean_matrix
def plot_result(T, GT, FT_target):
    """
    对矩阵T和GT进行可视化，其中GT是一个近似置换矩阵。
    GT按照每行第一个1的位置进行排序，T根据这个排序进行相同的行排序。

    参数:
    T (np.array): 输入的矩阵T。
    GT (np.array): 输入的近似置换矩阵GT。

    返回:
    save_path (str): 图像保存路径。
    """
    # 找到GT每行第一个1的位置
    first_one_indices = np.argmax(GT, axis=1)

    # 根据GT的排序对T进行排序
    sorted_T = T[np.argsort(first_one_indices)]

    # 根据GT的排序对GT进行排序
    sorted_GT = GT[np.argsort(first_one_indices)]

    # 根据GT的排序对GT进行排序
    sorted_FT = FT_target[np.argsort(first_one_indices)]

    # 绘制图像
    plt.figure(figsize=(18, 6), dpi=300)

    # 绘制矩阵 T
    plt.subplot(1, 4, 1)
    plt.imshow(sorted_T, cmap='viridis')
    plt.colorbar()
    plt.title('Matrix T (Sorted)')

    # 绘制矩阵 GT
    plt.subplot(1, 4, 2)
    plt.imshow(clean_T(sorted_T,1), cmap='viridis')
    plt.colorbar()
    plt.title('Matrix Clean T (Sorted)')

    # 绘制原始矩阵 GT
    plt.subplot(1, 4, 3)
    plt.imshow(sorted_GT, cmap='viridis')
    plt.colorbar()
    plt.title('Matrix GT (Sorted)')

    # 绘制原始矩阵 GT
    plt.subplot(1, 4, 4)
    plt.imshow(sorted_FT @ clean_T(sorted_T,1), cmap='viridis')
    plt.colorbar()
    plt.title('Matrix FT (Sorted)')

    # 保存图像
    save_path = os.path.join('debug', 'matrices_visualization.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    plt.show()

    return save_path


def matmul_torch(T, source_data):
    """
    使用 PyTorch 进行矩阵乘法加速。

    参数:
    T (np.array): 形状为 (64, 62) 的矩阵。
    source_data (np.array): 形状为 (10000, 62, 400) 的矩阵。

    返回:
    np.array: 加速后的矩阵乘法结果。
    """
    # 将 NumPy 数组转换为 PyTorch 张量
    T_torch = torch.from_numpy(T).float()
    source_data_torch = torch.from_numpy(source_data).float()

    # 检查是否有可用的 GPU
    if torch.cuda.is_available():
        T_torch = T_torch.cuda()
        source_data_torch = source_data_torch.cuda()

    # 执行矩阵乘法
    result_torch = torch.matmul(T_torch, source_data_torch)

    # 将结果转换回 NumPy 数组
    result_numpy = result_torch.cpu().numpy()

    return result_numpy


def random_permutation_matrix(n):
    # 生成一个 n x n 的单位矩阵
    permutation_matrix = np.eye(n)

    # 随机打乱矩阵的行
    np.random.shuffle(permutation_matrix)

    return permutation_matrix