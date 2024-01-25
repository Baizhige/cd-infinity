import torch.nn as nn
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from torch.optim.lr_scheduler import _LRScheduler
def EEG_wearing_transform(inputEEG,d,sigma=0.1,channel_number=64,max_bias=0.5,min_bias=-0.5):
    rand=torch.normal(0, sigma, (channel_number, channel_number))
    # rand = torch.clamp(rand, min_bias, max_bias)
    T = rand*d
    for i in range(channel_number):
        T[i,i]=1
    output=torch.mm(T,torch.squeeze(inputEEG))
    return torch.unsqueeze(output,dim=0)

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix
def channel_norm(input, channel=64):
    '''

    :param input: [1 64 256]
    :return:
    '''
    epsilon=0.0001
    temp_trial_ch_mean = torch.mean(input, dim=2).view(1,channel,1)
    temp_trial_ch_std = torch.std(input, dim=2).view(1,channel,1)
    A=torch.sub(input ,temp_trial_ch_mean)
    B=temp_trial_ch_std+epsilon
    out = torch.div(A, B)
    return out

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, mu, alpha, beta, total_steps, last_epoch=-1, verbose=False):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.total_steps = total_steps
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # 当前的训练进度 p 是当前步骤数除以总步骤数
        p = self.last_epoch / self.total_steps
        lr = [self.mu / (1 + self.alpha * p) ** self.beta for base_lr in self.base_lrs]
        return lr


def custom_ReLU_loss(d, d0=0.3, scale=3.0):
    # 使用ReLU函数来定义损失
    loss = torch.relu(d - d0)

    # 可选：将损失按比例缩放
    loss *= scale

    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算


def cov_loss(tensorA, tensorB):
    """
    计算两组 tensor（tensorA 和 tensorB）的平均协方差矩阵之间的 L2 距离。

    tensorA 和 tensorB 都是形状为 (batchsize, 1, c, T) 的 tensor。
    每个 tensor 包含 batchsize 个样本，每个样本是一个 c 行 T 列的矩阵。
    """

    def compute_mean_covariance(input_tensor):
        # 删除大小为 1 的维度，使 tensor 形状变为 (batchsize, c, T)
        input_tensor = input_tensor.squeeze(1)
        norms = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        input_tensor_normalized = input_tensor / norms
        covariance_matrices = torch.matmul(input_tensor_normalized, input_tensor_normalized.transpose(1, 2))
        # 计算平均协方差矩阵
        mean_covariance = covariance_matrices.mean(dim=0)
        return mean_covariance.squeeze()

    # 计算两个 tensor 的平均协方差矩阵
    mean_covariance_A = compute_mean_covariance(tensorA)
    mean_covariance_B = compute_mean_covariance(tensorB)

    # 计算 L2 距离作为损失
    loss = torch.norm(mean_covariance_A - mean_covariance_B, p=2)
    return loss


def generate_normalized_tensor(n_rows, n_columns):
    """
    Generates a torch.tensor of shape (n_rows, n_columns) with each row summing up to 1.

    :param n_rows: Number of rows in the tensor
    :param n_columns: Number of columns in the tensor
    :return: A torch.tensor of shape (n_rows, n_columns)
    """
    # Randomly generate a tensor
    random_tensor = torch.ones(n_rows, n_columns)

    # Normalize each row to sum up to 1
    row_sums = random_tensor.sum(dim=1).unsqueeze(1)
    normalized_tensor = random_tensor / row_sums

    return normalized_tensor

def generate_normalized_tensor_eye(n_rows, n_columns):
    """
    Generates a torch.tensor of shape (n_rows, n_columns) where the top-left corner
    is an identity matrix of size min(n_rows, n_columns). For rows not covered by
    the identity matrix, one element is set to 1, ensuring each row sums up to 1.

    :param n_rows: Number of rows in the tensor
    :param n_columns: Number of columns in the tensor
    :return: A torch.tensor of shape (n_rows, n_columns)
    """
    # Initialize tensor with zeros
    tensor = torch.zeros(n_rows, n_columns)

    # Set the top-left corner to an identity matrix
    min_dim = min(n_rows, n_columns)
    tensor[:min_dim, :min_dim] = torch.eye(min_dim)

    # Fill the remaining rows
    for i in range(min_dim, n_rows):
        # If identity matrix does not cover all columns, set the first available column to 1
        if min_dim < n_columns:
            tensor[i, min_dim] = 1
        # If identity matrix covers all columns, set the last column to 1
        else:
            tensor[i, -1] = 1

    return tensor


def observation_loss(T_N_A, T_N_B, T_F_A_R, T_F_B_R, T_F_A_GT_inv, T_F_B_GT):
    '''
    Args:
        T_N_A: source alignment head学习到的变换矩阵
        T_N_B: target alignment head学习到的变换矩阵
        T_F_A_R: source alignment head先验变换矩阵（RANDOM）
        T_F_B_R: target alignment head先验变换矩阵（RANDOM）
        T_F_A_GT_inv: source alignment head先验变换矩阵（GT）#为了更快计算，直接传入其逆矩阵
        T_F_B_GT: target alignment head先验变换矩阵（GT）
        derive：
        GT:
        I @ T_F_A_GT @ X_A = I @ T_F_B_GT @ X_B ---(1)
        if T_F_A_GT is invitable, then:
        X_A = inv(T_F_A_GT) @ T_F_B_GT @ X_B  ----(2)

        RADNOM：
        T_N_A @ T_F_A_R @ X_A = T_N_B @ T_F_B_R @ X_B ---(3)
        Merge (2) into (3), then:
        T_N_A @ T_F_A_R @ inv(T_F_A_GT) @ T_F_B_GT @ X_B =  T_N_B @ T_F_B_R @ X_B
        then:
        T_N_A @ T_F_A_R @ inv(T_F_A_GT) @ T_F_B_GT =  T_N_B @ T_F_B_R

        we have loss function:

        loss = L2(T_N_A @ T_F_A_R @ inv(T_F_A_GT) @ T_F_B_GT - T_N_B @ T_F_B_R) -> 0

    Returns:
        loss
    '''
    difference = T_N_A @ T_F_A_R @ T_F_A_GT_inv @ T_F_B_GT - T_N_B @ T_F_B_R
    return torch.norm(difference)


def print_gradients(model):
    """
    打印 nn.Module 所有参数的梯度。

    :param model: nn.Module 的实例
    """
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(f"Gradient of {name}: {parameter.grad}")
        else:
            print(f"Gradient of {name}: None (not computed or no gradient)")

def cholesky_decomposition(S_a, S_b):
    """
    Solves for T using Cholesky decomposition.
    Assumes that C (S_a^{-1}S_b) is positive definite.
    """
    # Calculate C = S_a^{-1}S_b
    C = torch.inverse(S_a) @ S_b

    # Cholesky decomposition of C
    L = torch.cholesky(C)

    return L

def eigenvalue_decomposition(S_a, S_b):
    """
    Solves for T using eigenvalue decomposition.
    """
    # Calculate C = S_a^{-1}S_b
    C = torch.inverse(S_a) @ S_b

    # Eigenvalue decomposition of C
    eigenvalues, eigenvectors = torch.eig(C)

    # Check for negative eigenvalues which cannot be handled in this context
    if torch.any(eigenvalues < 0):
        raise ValueError("C has negative eigenvalues, cannot compute sqrt for eigenvalue decomposition.")

    # Construct T
    D_sqrt = torch.diag(torch.sqrt(eigenvalues))
    T = eigenvectors @ D_sqrt

    return T


if __name__=="__main__":
    d = np.load(os.path.join('.', 'config', 'd.npy'))
    inputEEG = np.load(os.path.join('..', 'EEGData', 'eeg-motor-movementimagery-dataset-1.0.0','processedData','MI','Left','S001R04_11.npy'),allow_pickle=True)
    inputEEG = torch.tensor(np.float64(inputEEG).reshape([1,64,256]))
    EEG_wearing_transform(inputEEG,d=d)