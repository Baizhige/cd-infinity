import numpy as np
import matplotlib.pyplot as plt
import os

def match_histograms(source, template, bins=256):
    """
    将矩阵source的直方图匹配到矩阵template的直方图。

    参数:
    source (np.array): 需要调整分布的矩阵。
    template (np.array): 目标分布的矩阵。
    bins (int): 用于计算直方图的bins数量。

    返回:
    matched (np.array): 分布调整后的source矩阵。
    """
    # 计算两个矩阵的直方图和累积分布函数
    source_hist, source_bin_edges = np.histogram(source, bins=bins, range=[np.min(source), np.max(source)], density=True)
    source_cdf = np.cumsum(source_hist) / np.sum(source_hist)
    template_hist, template_bin_edges = np.histogram(template, bins=bins, range=[np.min(template), np.max(template)], density=True)
    template_cdf = np.cumsum(template_hist) / np.sum(template_hist)

    # 创建映射函数
    interp_map = np.interp(source_cdf, template_cdf, template_bin_edges[:-1])

    # 将映射函数应用到原矩阵
    source_shape = source.shape
    source_flatten = source.ravel()
    source_matched_flatten = np.interp(source_flatten, source_bin_edges[:-1], interp_map)
    matched = source_matched_flatten.reshape(source_shape)

    return matched

def histogram_matching_and_plot(A, B, bins=256):
    """
    对矩阵A和B进行直方图匹配，并绘制处理前后的直方图。

    参数:
    A, B (np.array): 输入的两个矩阵。
    bins (int): 用于计算直方图的bins数量。

    返回:
    A_matched, B_matched (np.array): 直方图匹配后的矩阵。
    """
    # 匹配直方图
    A_matched = match_histograms(A, B, bins)
    B_matched = match_histograms(B, A, bins)

    # 绘制直方图
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].hist(A.ravel(), bins=bins, color='blue', alpha=0.7)
    axs[0, 0].set_title("Original Histogram of A")

    axs[0, 1].hist(B.ravel(), bins=bins, color='green', alpha=0.7)
    axs[0, 1].set_title("Original Histogram of B")

    axs[1, 0].hist(A_matched.ravel(), bins=bins, color='blue', alpha=0.7)
    axs[1, 0].set_title("Matched Histogram of A")

    axs[1, 1].hist(B_matched.ravel(), bins=bins, color='green', alpha=0.7)
    axs[1, 1].set_title("Matched Histogram of B")

    # 保存图像
    plt.tight_layout()
    save_path = os.path.join("debug", "matrix_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    return A_matched, B_matched, save_path



def plot_matching_result(A, B, A_matched, B_matched, bins=128):
    """
    绘制A B C D四个矩阵的分布

    参数:
    A, B (np.array): 输入的两个矩阵。
    bins (int): 用于计算直方图的bins数量。

    返回:
    A_matched, B_matched (np.array): 直方图匹配后的矩阵。
    """

    # 绘制直方图
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].hist(A.ravel(), bins=bins, color='blue', alpha=0.7)
    axs[0, 0].set_title("Original Histogram of A")

    axs[0, 1].hist(B.ravel(), bins=bins, color='green', alpha=0.7)
    axs[0, 1].set_title("Original Histogram of B")

    axs[1, 0].hist(A_matched.ravel(), bins=bins, color='blue', alpha=0.7)
    axs[1, 0].set_title("Matched Histogram of A")

    axs[1, 1].hist(B_matched.ravel(), bins=bins, color='green', alpha=0.7)
    axs[1, 1].set_title("Matched Histogram of B")

    # 保存图像
    plt.tight_layout()
    save_path = os.path.join("debug", "matrix_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)



    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 绘制矩阵 A
    axs[0, 0].imshow(A, cmap='viridis')
    axs[0, 0].set_title("Matrix A")

    # 绘制矩阵 B
    axs[0, 1].imshow(B, cmap='viridis')
    axs[0, 1].set_title("Matrix B")

    # 绘制矩阵 A_matched
    axs[1, 0].imshow(A_matched, cmap='viridis')
    axs[1, 0].set_title("Matrix A_matched")

    # 绘制矩阵 B_matched
    axs[1, 1].imshow(B_matched, cmap='viridis')
    axs[1, 1].set_title("Matrix B_matched")

    # 保存图像
    plt.tight_layout()
    save_path = os.path.join("debug", "matrices_visualization.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    plt.show()



    return A_matched, B_matched, save_path



if __name__ =="__main__":
    # 再次使用不同分布的示例矩阵进行测试
    fake_avg_cov_matrix_reg = np.load(os.path.join("debug","dataExp3_cov_matrix.npy"))
    avg_cov_matrix_reg = np.load(os.path.join("debug","dataExp12_cov_matrix.npy"))

    # 调用函数
    A_matched, B_matched, save_path = histogram_matching_and_plot(fake_avg_cov_matrix_reg, avg_cov_matrix_reg, bins=128)
    np.save(os.path.join("debug","matched_dataExp3_cov_matrix.npy"),A_matched)
    np.save(os.path.join("debug","matched_dataExp12_cov_matrix.npy"),B_matched)

