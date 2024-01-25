import numpy as np
import warnings
from sheet3 import match_histograms, plot_matching_result
from sheet2 import generate_naive_T,plot_result, matmul_torch,random_permutation_matrix, clean_T

def compute_average_covariance_matrix(X, z_threshold=10, norm_mode='z_score', reg_mode='regularization', lambda_val=None):
    """
    计算平均协方差矩阵。

    参数:
    X (numpy.array): 原始数据集，形状为 (n_samples, c, T)。
    z_threshold (float, 可选): 用于Z分数异常值筛除的阈值，默认为 2。
    norm_mode (str, 可选): 选择归一化方法，可选 'min_max'，'z_score' 或 'unit_length'，默认为 'unit_length'。
    reg_mode (str, 可选): 选择提高可逆性的方法，可选 'pseudo_inverse' 或 'regularization'，默认为 'regularization'。
    lambda_val (float, 可选): 正则化中对角线上的值，如果为 None，则自适应选择，默认为 None。

    返回:
    tuple: (平均协方差矩阵, 正则化后的平均协方差矩阵)
    """

    def z_score_normalization(data):
        """Z分数归一化"""
        return (data - np.mean(data, axis=2, keepdims=True)) / np.std(data, axis=2, keepdims=True)

    def min_max_normalization(data):
        """最小-最大归一化"""
        return (data - np.min(data, axis=2, keepdims=True)) / (np.max(data, axis=2, keepdims=True) - np.min(data, axis=2, keepdims=True))

    def unit_length_normalization(data):
        """单位长度归一化"""
        norm = np.linalg.norm(data, axis=2, keepdims=True)
        norm[norm == 0] = 1  # 避免除以0
        return data / norm

    # Z分数异常值筛除
    Z = np.abs((X - np.mean(X, axis=2, keepdims=True)) / np.std(X, axis=2, keepdims=True))
    original_size = X.shape[0]
    X = X[np.all(Z < z_threshold, axis=(1, 2))]
    print(f"Z分数方法筛除了 {original_size - X.shape[0]} 个样本，剩余 {X.shape[0]} 个样本。")

    # 数据归一化
    if norm_mode not in ['min_max', 'z_score', 'unit_length']:
        warnings.warn("未知的归一化模式，使用默认的单位长度归一化。")
        norm_mode = 'no_norm'

    if norm_mode == 'min_max':
        X_norm = min_max_normalization(X)
        print("使用最小-最大归一化方法。")
    elif norm_mode == 'z_score':
        X_norm = z_score_normalization(X)
        print("使用Z分数归一化方法。")
    elif norm_mode == 'unit_length':
        X_norm = unit_length_normalization(X)
        print("使用单位长度归一化方法。")
    else:
        X_norm = X
        print("不使用归一化")

    # 计算平均协方差矩阵
    cov_matrices = [np.cov(x.T, rowvar=False) for x in X_norm]
    avg_cov_matrix = np.mean(cov_matrices, axis=0)

    # 检查协方差矩阵的可逆性
    if np.linalg.matrix_rank(avg_cov_matrix) == avg_cov_matrix.shape[0]:
        print("协方差矩阵是可逆的。")
        return avg_cov_matrix, avg_cov_matrix

    # 提高协方差矩阵可逆性
    if reg_mode not in ['pseudo_inverse', 'regularization']:
        warnings.warn("未知的正则化模式，使用默认的正则化方法。")
        reg_mode = 'regularization'

    if reg_mode == 'pseudo_inverse':
        avg_cov_matrix_reg = np.linalg.pinv(avg_cov_matrix)
        print("使用伪逆方法提高协方差矩阵的可逆性。")
    else:  # 默认使用正则化方法
        if lambda_val is None:
            # 自适应选择 lambda
            lambda_val = 1e-6
            while np.linalg.matrix_rank(avg_cov_matrix + lambda_val * np.eye(avg_cov_matrix.shape[0])) < avg_cov_matrix.shape[0]:
                lambda_val *= 10
        avg_cov_matrix_reg = avg_cov_matrix + lambda_val * np.eye(avg_cov_matrix.shape[0])
        print(f"使用正则化方法提高协方差矩阵的可逆性。选取的 lambda 为 {lambda_val}。")

    return avg_cov_matrix, avg_cov_matrix_reg


def solve_T(S_a, S_b):
    """
    求解置换矩阵 T，使得 S_b = TS_aT^T 成立。

    参数:
    S_a (numpy.array): 平均协方差矩阵 S_a，形状为 c x c。
    S_b (numpy.array): 平均协方差矩阵 S_b，形状为 c x c。

    返回:
    numpy.array: 求解得到的置换矩阵 T。

    功能:
    此函数接收两个平均协方差矩阵 S_a 和 S_b，通过特征分解的方法寻找一个置换矩阵 T，
    使得 S_b = TS_aT^T 成立。此函数使用了 numpy 库进行矩阵操作和特征分解，
    并检查传入的矩阵是否可逆和维度是否匹配。
    """

    # 检查 S_a 和 S_b 的维度是否匹配
    if S_a.shape != S_b.shape:
        raise ValueError("S_a 和 S_b 的维度必须匹配。")

    # 检查 S_a 和 S_b 是否可逆
    if np.linalg.det(S_a) == 0 or np.linalg.det(S_b) == 0:
        raise ValueError("S_a 和 S_b 必须是可逆的。")
    # 对 S_a 和 S_b 进行特征分解

    eigvals_a, eigvecs_a = np.linalg.eigh(S_a)
    eigvals_b, eigvecs_b = np.linalg.eigh(S_b)

    # 排序特征值和对应的特征向量
    idx_a = eigvals_a.argsort()[::-1]
    eigvals_a = eigvals_a[idx_a]
    eigvecs_a = eigvecs_a[:, idx_a]

    idx_b = eigvals_b.argsort()[::-1]
    eigvals_b = eigvals_b[idx_b]
    eigvecs_b = eigvecs_b[:, idx_b]

    # 构造置换矩阵 T
    T = eigvecs_b @ np.linalg.inv(eigvecs_a)

    return T


def solve_T_modified(S_a, S_b, k=1):
    """
    求解置换矩阵 T，使得 S_b = TS_aT^T 成立，通过保留前k个最大特征值的方式。

    参数:
    S_a (numpy.array): 平均协方差矩阵 S_a，形状为 c x c。
    S_b (numpy.array): 平均协方差矩阵 S_b，形状为 c x c。
    k (int): 保留的特征值数量。

    返回:
    numpy.array: 求解得到的置换矩阵 T。

    功能:
    此函数接收两个平均协方差矩阵 S_a 和 S_b，通过特征分解的方法寻找一个置换矩阵 T，
    使得 S_b = TS_aT^T 成立。在这个过程中，只保留每个协方差矩阵前k个最大的特征值
    及其对应的特征向量。
    """

    # 检查 S_a 和 S_b 的维度是否匹配
    if S_a.shape != S_b.shape:
        raise ValueError("S_a 和 S_b 的维度必须匹配。")

    # 对 S_a 和 S_b 进行特征分解
    eigvals_a, eigvecs_a = np.linalg.eigh(S_a)
    eigvals_b, eigvecs_b = np.linalg.eigh(S_b)

    # 仅保留前k个最大特征值对应的特征向量
    idx_a = np.argsort(eigvals_a)[::-1][:k]
    idx_b = np.argsort(eigvals_b)[::-1][:k]
    eigvecs_a = eigvecs_a[:, idx_a]
    eigvecs_b = eigvecs_b[:, idx_b]

    # 计算置换矩阵 T，使用伪逆处理
    T = eigvecs_b @ np.linalg.pinv(eigvecs_a)

    return T


import os
print("loading data")
source_data = np.load(os.path.join("..", "..", "EEGData", "dataset_MengData", "concatedData", "all",
                                                "dataExp12_128_T62_None_default.npy"))


target_data = np.load(os.path.join("..", "..", "EEGData", "dataset_MengData", "concatedData", "all",
                                                "dataExp3_128_T64_None_default.npy"))


source_data = matmul_torch(generate_naive_T(64,62), source_data)

GT = np.load(os.path.join('config','transformation_matrix_MengExp12_MengExp3_(64_62)_new.npy'))

FT_target = np.load(os.path.join('debug','MengDataE3FlipMatrix.npy'))

print("computing average cov martix")
source_avg_cov_matrix, source_avg_cov_matrix_reg = compute_average_covariance_matrix(source_data)
print("computing average cov martix")
target_avg_cov_matrix, target_avg_cov_matrix_reg = compute_average_covariance_matrix(target_data)

print("matching cov martix")
# source_avg_cov_matrix_reg_matched = match_histograms(source_avg_cov_matrix_reg, target_avg_cov_matrix_reg, bins=64)
source_avg_cov_matrix_reg_matched = source_avg_cov_matrix_reg.copy()
target_avg_cov_matrix_reg_matched = target_avg_cov_matrix_reg.copy()

plot_matching_result(source_avg_cov_matrix_reg, target_avg_cov_matrix_reg, source_avg_cov_matrix_reg_matched, target_avg_cov_matrix_reg_matched, bins=64)

print("computing T")
T = solve_T(source_avg_cov_matrix_reg_matched, target_avg_cov_matrix_reg)
# T = T / T.sum(axis=1, keepdims=True)
# print(T.sum(axis=1, keepdims=True))
# np.save(os.path.join("debug","transformation_matrix_MengExp12_MengExp3_(64_64)_without_prior.npy"), T)
print("ploting result T")
plot_result(T, GT, FT_target)
print("ploting result cov matrix")
source_data_transformed = matmul_torch(T, source_data)

source_avg_cov_matrix_transformed, source_avg_cov_matrix_reg_transformed = compute_average_covariance_matrix(source_data_transformed)

plot_matching_result(source_avg_cov_matrix_reg_matched, target_avg_cov_matrix_reg_matched, source_avg_cov_matrix_transformed, target_avg_cov_matrix_reg_matched, bins=64)

