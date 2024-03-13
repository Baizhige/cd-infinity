import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# 方法列表
# methods_list = ["Baseline", "DDC", "DeepCoral", "DANN", "DANNWass", "EEG_Infinity003", "EEG_Infinity004", "EEG_Infinity005Wass"]
methods_list = ["Baseline", "EEG_Infinity005Wass"]
# 初始化存储所有特征的列表
features_all = []

# 加载所有特征
for method in methods_list:
    for suffix in ['s_FC_features_stacked', 't_FC_features_stacked']:
        path = os.path.join("record", f"{method}_{suffix}.pth")
        features = torch.load(path)
        features_all.append(features)

# 合并所有特征并进行T-SNE降维
features_combined = torch.cat(features_all, dim=0).detach().numpy()
tsne = TSNE(n_components=2, random_state=42)
features_reduced = tsne.fit_transform(features_combined)

# 分割降维后的特征以便绘图
features_reduced_split = []
start = 0
for features in features_all:
    end = start + features.shape[0]
    features_reduced_split.append(features_reduced[start:end, :])
    start = end

# 绘制散点图，2行4列的子图
fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=300)

# 定义每组特征的颜色
colors = ['r', 'g']  # 第一种颜色用于s_FC_features_stacked，第二种颜色用于t_FC_features_stacked

for i, ax in enumerate(axes.flatten()):
    if i < len(methods_list):
        # 选择当前子图对应的降维后的特征
        features_set_s = features_reduced_split[i * 2]
        features_set_t = features_reduced_split[i * 2 + 1]

        # 绘制当前子图
        ax.scatter(features_set_s[:, 0], features_set_s[:, 1], color=colors[0], label=f'{methods_list[i]} Source')
        ax.scatter(features_set_t[:, 0], features_set_t[:, 1], color=colors[1], label=f'{methods_list[i]} Target')
        ax.set_title(methods_list[i])
        ax.legend()

# 调整布局和保存
plt.tight_layout()
plt.suptitle('T-SNE Reduction to 2 Dimensions for Various Methods', y=1.05)
plt.savefig(os.path.join('figures','features_tsne_comparison.pdf'), format='pdf', dpi=300)
