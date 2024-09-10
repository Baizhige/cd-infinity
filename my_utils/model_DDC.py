import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

# def wasserstein_distance(source_data, target_data, reg=1e-1):
#     batch_size, num_features = source_data.shape
#     source_data_expanded = source_data.unsqueeze(1).repeat(1, batch_size, 1)
#     target_data_expanded = target_data.unsqueeze(0).repeat(batch_size, 1, 1)
#     pairwise_distances = torch.cdist(source_data_expanded, target_data_expanded, p=2)
#
#     # No need to index ot.sinkhorn2's result
#     wasserstein_distances = torch.tensor(
#         [ot.sinkhorn2([], [], pairwise_distances[i].detach().cpu().numpy(), reg=reg) for i in range(batch_size)]
#     )
#     return wasserstein_distances
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        Convert the source domain data and target domain data into kernel matrices, as mentioned above in K.
        Params:
            source: Source domain data (n * len(x))
            target: Target domain data (m * len(y))
            kernel_mul:
            kernel_num: The number of different Gaussian kernels
            fix_sigma: Sigma values for different Gaussian kernels
        Return:
            sum(kernel_val): The sum of multiple kernel matrices
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# Calculate the number of rows in the matrix. Typically, the source and target have the same dimensions, which makes the computation easier.
    total = torch.cat([source, target], dim=0)# Concatenate the source and target along the column direction
    # Replicate the total (n+m) times
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # Replicate each row of total into (n+m) rows, i.e., extend each data point into (n+m) copies
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # Calculate the sum between any two data points. In the resulting matrix, coordinate (i, j) represents the L2 distance between the i-th and j-th rows of total (i == j is 0).
    L2_distance = ((total0-total1)**2).sum(2)
    # Adjust the sigma value for the Gaussian kernel function

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # Use fix_sigma as the median and take kernel_num bandwidth values as multiples of kernel_mul (for example, if fix_sigma is 1, we get [0.25, 0.5, 1, 2, 4])
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # Mathematical expression of the Gaussian kernel function
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # Obtain the final kernel matrix
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        Calculate the MMD distance between the source domain data and target domain data.
        Params:
            source: Source domain data (n * len(x))
            target: Target domain data (m * len(y))
            kernel_mul:
            kernel_num: The number of different Gaussian kernels
            fix_sigma: Sigma values for different Gaussian kernels
        Return:
            loss: MMD loss
    '''
    batch_size = int(source.size()[0])# Usually, the batch size of the source and target domains is assumed to be the same
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # Divide the kernel matrix into 4 parts according to equation (3)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss# Since n == m in most cases, the L matrix is generally not included in the computation


class depthwise_separable_conv(nn.Module):  # Depthwise separable convolution
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DDC(nn.Module):  # Net3: DDC using EEGNet
    def __init__(self,transfer_matrix):
        super(DDC, self).__init__()
        self.kernel_size = 128
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 64
        self.num_classes = 2
        self.feature_map_size = 192
        self.transfer_matrix = transfer_matrix
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('fb-1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('fc-2', nn.Linear(128, 64))
        self.class_classifier.add_module('fb-2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('fc-3', nn.Linear(64, self.num_classes))

    def ori_forward(self, source_data, target_data):
        inter_global_loss = 0
        inter_ele_loss = torch.from_numpy(np.array(0)).cuda()
        intra_ele_loss = torch.from_numpy(np.array(0)).cuda()
        source_all = self.feature(source_data)
        source_all = source_all.view(-1, self.feature_map_size)

        if self.training:
            target_all = self.feature(target_data)
            target_all = target_all.view(-1, self.feature_map_size)
            inter_global_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)
            output = self.class_classifier(source_all)
        return output, inter_global_loss, intra_ele_loss, inter_ele_loss


    def forward(self, input_data, domain=None, alpha=None):
        if domain == 1:
            # target domain transfer
            input_data = torch.matmul(self.transfer_matrix, input_data)
        input_data = input_data.to(torch.float32)
        feature = self.feature(input_data)
        feature = feature.view(-1, self.feature_map_size)
        output = self.class_classifier(feature)
        return output, feature, None, None


class DeepCoral(nn.Module):  # Net4: DeepCoral
    def __init__(self):
        super(DeepCoral, self).__init__()
        self.kernel_size = 128
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 64
        self.num_classes = 2
        self.feature_map_size = 192
        self.feature = nn.Sequential()

        self.feature.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('p-1', nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.feature.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('b-2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('e-1', nn.ELU())

        self.feature.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('d-1', nn.Dropout(p=0.25))

        self.feature.add_module('c-3', depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('p-2', nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('b-3', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('e-2', nn.ELU())
        self.feature.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('d-2', nn.Dropout(p=0.25))

        self.classifier_1 = nn.Sequential()
        self.classifier_1.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.classifier_1.add_module('fb-1', nn.BatchNorm1d(128))

        self.classifier_2 = nn.Sequential()
        self.classifier_2.add_module('fc-2', nn.Linear(128, 64))
        self.classifier_2.add_module('fb-2', nn.BatchNorm1d(64))

        self.classifier_3 = nn.Sequential()
        self.classifier_3.add_module('fc-3', nn.Linear(64, self.num_classes))

    def forward(self, source_data, target_data):

        loss_1 = 0
        loss_2 = 0
        loss_3 = 0

        source_all = self.feature(source_data)
        source_all_1 = source_all.view(-1, self.feature_map_size)
        source_all_2 = self.classifier_1(source_all_1)
        source_all_3 = self.classifier_2(source_all_2)

        if self.training:
            target_all = self.feature(target_data)

            target_all_1 = target_all.view(-1, self.feature_map_size)
            s1 = torch.matmul(source_all_1.T, source_all_1)
            t1 = torch.matmul(target_all_1.T, target_all_1)
            loss_1 += euclidean_dist(s1, t1)

            target_all_2 = self.classifier_1(target_all_1)
            s2 = torch.matmul(source_all_2.T, source_all_2)
            t2 = torch.matmul(target_all_2.T, target_all_2)
            loss_2 += euclidean_dist(s2, t2)

            target_all_3 = self.classifier_2(target_all_2)
            s3 = torch.matmul(source_all_3.T, source_all_3)
            t3 = torch.matmul(target_all_3.T, target_all_3)
            loss_3 += euclidean_dist(s3, t3)

        output = self.classifier_3(source_all_3)
        return output, loss_1, loss_2, loss_3