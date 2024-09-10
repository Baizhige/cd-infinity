import torch.nn as nn
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from torch.optim.lr_scheduler import _LRScheduler
def EEG_wearing_transform(inputEEG,d,sigma=0.1,channel_number=64,max_bias=0.5,min_bias=-0.5):
    rand=torch.normal(0, sigma, (channel_number, channel_number))
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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : The calculated value of the confusion matrix
    - classes : The columns corresponding to each row and column in the confusion matrix
    - normalize : True: display percentage, False: display number
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

    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

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
        p = self.last_epoch / self.total_steps
        lr = [self.mu / (1 + self.alpha * p) ** self.beta for base_lr in self.base_lrs]
        return lr


def custom_ReLU_loss(d, d0=0.3, scale=3.0):

    loss = torch.relu(d - d0)

    loss *= scale

    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Convert source domain data and target domain data into kernel matrix, i.e. K in the above text
    Params:
    source: source domain data (n * len(x))
    target: target domain data (m * len(y))
    kernel_mul:
    kernel_num: the number of different Gaussian kernels
    fix_sigma: sigma value of different Gaussian kernels
    Return:
    sum(kernel_val): the sum of multiple kernel matrices
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# Find the number of rows in the matrix. Generally, the scale of source and target is the same, which is convenient for calculation
    total = torch.cat([source, target], dim=0) # Merge source and target in column direction
    # Copy total (n+m) copies
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # Copy each row of total into (n+m) rows, that is, each data is expanded into (n+m) copies
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # Find the sum of any two data points. The coordinates (i, j) in the resulting matrix represent the l2 distance between the i-th row of data and the j-th row of data in total (0 when i==j)
    L2_distance = ((total0-total1)**2).sum(2)
    # Adjust the sigma value of the Gaussian kernel function
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # Take fix_sigma as the median value and kernel_mul as the multiple to get kernel_num bandwidth values ​​(for example, when fix_sigma is 1, we get [0.25, 0.5, 1, 2, 4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # Mathematical expression of Gaussian kernel function
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # Get the final kernel matrix
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Calculate the MMD distance between source domain data and target domain data
    Params:
    source: source domain data (n * len(x))
    target: target domain data (m * len(y))
    kernel_mul:
    kernel_num: the number of different Gaussian kernels
    fix_sigma: sigma values of different Gaussian kernels
    Return:
    loss: MMD loss
    '''
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def cov_loss(tensorA, tensorB):
    """
    Calculate the L2 distance between the mean covariance matrices of two sets of tensors (tensorA and tensorB).

    tensorA and tensorB are both tensors of shape (batchsize, 1, c, T).
    Each tensor contains batchsize samples, and each sample is a matrix with c rows and T columns.
    """

    def compute_mean_covariance(input_tensor):
        # Delete the dimension of size 1 to make the tensor shape become (batchsize, c, T)
        input_tensor = input_tensor.squeeze(1)
        norms = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        input_tensor_normalized = input_tensor / norms
        covariance_matrices = torch.matmul(input_tensor_normalized, input_tensor_normalized.transpose(1, 2))

        mean_covariance = covariance_matrices.mean(dim=0)
        return mean_covariance.squeeze()

    mean_covariance_A = compute_mean_covariance(tensorA)
    mean_covariance_B = compute_mean_covariance(tensorB)

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
    Print the gradients of all parameters of nn.Module.

    :param model: instance of nn.Module
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