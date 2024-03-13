import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# 方法列表
# methods_list = ["Baseline", "DDC", "DeepCoral", "DANN", "DANNWass", "EEG_Infinity003", "EEG_Infinity004", "EEG_Infinity005Wass"]
methods_list = ["Baseline", "EEG_Infinity005Wass"]

# 绘制散点图，2行4列的子图
fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)

# 遍历每个方法，对source和target一起进行T-SNE降维
for i, method in enumerate(methods_list):
    # 计算子图的位置
    ax = axes[i // 4, i % 4]

    # 加载source和target特征
    source_path = os.path.join("record", f"{method}_s_FC_features_stacked.pth")
    target_path = os.path.join("record", f"{method}_t_FC_features_stacked.pth")
    features_s = torch.load(source_path)
    features_t = torch.load(target_path)

    # 合并source和target特征
    features_combined = torch.cat((features_s, features_t), dim=0).detach().numpy()

    # 对合并的特征进行T-SNE降维
    tsne = TSNE(n_components=2, random_state=42, early_exaggeration=12)
    features_combined_reduced = tsne.fit_transform(features_combined)

    # 分割降维后的特征，以便分别绘图
    split_index = features_s.shape[0]
    features_s_reduced = features_combined_reduced[:split_index, :]
    features_t_reduced = features_combined_reduced[split_index:, :]

    # 绘制当前子图
    ax.scatter(features_s_reduced[:, 0], features_s_reduced[:, 1], color='r', label=f'{method} Source', s=5)
    ax.scatter(features_t_reduced[:, 0], features_t_reduced[:, 1], color='g', label=f'{method} Target', s=5)
    ax.set_title(method)
    ax.legend()


# 调整布局和保存
plt.tight_layout()
plt.suptitle('Individual T-SNE Reduction for Source and Target Data', y=1.05)
plt.savefig(os.path.join('figures','features_tsne_comparison.pdf'), format='pdf', dpi=300)

