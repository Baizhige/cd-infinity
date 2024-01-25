import torch.nn as nn
import torch
from torch.autograd import Function
import scipy.io as scio
import os
import torch.nn.functional as F




class EEG_Infinity(nn.Module):

    def __init__(self, transfer_matrix_source, transfer_matrix_target, FIR_order=17, FIR_n=1):
        super(EEG_Infinity, self).__init__()
        self.feature_map_size = 192
        self.num_classes = 2
        self.num_channels = transfer_matrix_source.size()[0]

        # 定义了源域alignment heads
        self.alignment_head_source = Alignment_head(transfer_matrix=transfer_matrix_source,
                                                    FIR_order=FIR_order, FIR_n=FIR_n)
        self.alignment_head_target = Alignment_head(transfer_matrix=transfer_matrix_target,
                                                    FIR_order=FIR_order, FIR_n=FIR_n)
        # 冻结源域的 channel_transfer_matrix
        self.alignment_head_source.frozen_transfer_matrix()

        # ChannelNorm()层定义，防止BN层的均值和方差乱飘
        self.channel_norm = ChannelNorm()

        # 定义了特征提取器
        self.feature = Feature_endocer_EEGNet()
        # 定义了特征分类器
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(128, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, self.num_classes))
        # 定义了领域分类器
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc1', nn.Linear(self.feature_map_size, 128))
        self.domain_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.domain_classifier.add_module('c_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('c_drop1', nn.Dropout())
        self.domain_classifier.add_module('c_fc2', nn.Linear(128, 64))
        self.domain_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('c_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('c_fc3', nn.Linear(64, 2))

    def forward(self, input_data, domain, alpha):
        input_data = input_data.to(torch.float32)
        if domain == 0:
            filter_output, spatial_output = self.alignment_head_source(input_data)
        else:
            filter_output, spatial_output = self.alignment_head_target(input_data)

        _feature_ = self.feature(self.channel_norm(filter_output)).view(-1, self.feature_map_size)

        _reverse_feature_ = ReverseLayerF.apply(_feature_, alpha)

        class_output = self.class_classifier(_feature_)
        domain_output = self.domain_classifier(_reverse_feature_)

        return class_output, domain_output, filter_output, spatial_output


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ChannelNorm(nn.Module):
    def forward(self, input):
        batch_size, n_layers, n_channels, n_sampling = input.size()

        # Reshape input to (batch_size * n_layers, n_channels, n_sampling)
        input_reshaped = input.view(-1, n_channels, n_sampling)

        # Calculate mean and variance along the last dimension
        mean = torch.mean(input_reshaped, dim=-1, keepdim=True)
        variance = torch.var(input_reshaped, dim=-1, keepdim=True, unbiased=False)

        # Normalize input
        output = (input_reshaped - mean) / torch.sqrt(variance + 1e-8)

        # Reshape output back to the original shape
        output = output.view(batch_size, n_layers, n_channels, n_sampling)

        return output


class FIR_convolution(nn.Module):
    def __init__(self, FIR_n, FIR_order):
        super(FIR_convolution, self).__init__()
        # 创建一个2D卷积层，用于实现FIR滤波器
        self.FIR_order = FIR_order
        self.conv = nn.Conv2d(1, FIR_n, (1, FIR_order), padding=0, bias=False)

        # 初始化参数，避免过于极端的滤波效果
        self.initialize_parameters()

    def forward(self, x):
        # 添加零填充
        x_padded = F.pad(x, (int(self.conv.kernel_size[1] / 2), int(self.conv.kernel_size[1] / 2), 0, 0), mode='constant', value=0)
        # 应用卷积（使用归一化的权重）
        return F.conv2d(x_padded, self.conv.weight, padding=0)

    def initialize_parameters(self):
        """ 初始化滤波器的参数，使每个参数的值为 1/FIR_order """
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / self.FIR_order)
class Alignment_head(nn.Module):
    def __init__(self, transfer_matrix, FIR_order=17, FIR_n=1):
        super(Alignment_head, self).__init__()
        self.channel_transfer_matrix = nn.Parameter(torch.eye(transfer_matrix.size()[0]))
        self.channel_transfer_matrix_fixed = transfer_matrix.cuda()
        self.domain_filter = FIR_convolution(FIR_n, FIR_order)


    def forward(self, input_data):
        _input_data_ = input_data.to(torch.float32)
        _input_data_ = torch.matmul(self.channel_transfer_matrix_fixed, _input_data_)
        output_spatial = torch.matmul(self.channel_transfer_matrix, _input_data_)
        output_filter = self.domain_filter(output_spatial)
        return output_filter, output_spatial

    def custom_zero_grad(self):
        for param in [self.channel_transfer_matrix, self.domain_filter.conv.weight]:
            if param.grad is None:
                # 初始化梯度为零
                param.grad = torch.zeros_like(param.data)
            else:
                param.grad.zero_()


    def get_magnitude_loss(self, alpha=0.5):
        # 确保 channel_transfer_matrix 的每行之和为 1
        # row_sums = torch.sum(self.channel_transfer_matrix, dim=1)
        # loss_matrix = torch.sum((row_sums - 1) ** 2)

        # 确保 domain_filter 的每个卷积核参数之和为 1
        filter_sums = self.domain_filter.conv.weight.sum(dim=(1, 2, 3))
        loss_filter = torch.sum((filter_sums - 1) ** 2)

        # 返回两部分损失的总和
        return loss_filter*1

    def frozen_transfer_matrix(self):
        # 冻结 channel_transfer_matrix 参数
        self.channel_transfer_matrix.requires_grad = False
class Feature_endocer_EEGNet(nn.Module):
    def __init__(self):
        super(Feature_endocer_EEGNet, self).__init__()
        self.kernel_size = 64
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = 64
        self.feature = nn.Sequential()
        # (N,1,64,256)
        self.feature.add_module('f_conv1', nn.Conv2d(1, self.F1, (1, self.kernel_size), padding=0))
        self.feature.add_module('f_padding1',
                                nn.ZeroPad2d((int(self.kernel_size / 2) - 1, int(self.kernel_size / 2), 0, 0)))
        self.feature.add_module('f_batchnorm1', nn.BatchNorm2d(self.F1, False))
        # (N,F1,64,256)
        self.feature.add_module('f_conv2', nn.Conv2d(self.F1, self.F1 * self.D, (self.num_channel, 1), groups=8))
        self.feature.add_module('f_batchnorm2', nn.BatchNorm2d(self.F1 * self.D, False))
        self.feature.add_module('f_ELU2', nn.ELU())
        # (N,F1*D,1,256)
        self.feature.add_module('f_Pooling3', nn.AvgPool2d(kernel_size=(1, 4)))
        self.feature.add_module('f_dropout3', nn.Dropout(p=0.25))
        # (N,F1*D,1,256/4)
        self.feature.add_module('f_conv4',
                                depthwise_separable_conv(self.F1 * self.D, self.F2, int(self.kernel_size / 4)))
        self.feature.add_module('f_padding4',
                                nn.ZeroPad2d((int(self.kernel_size / 8) - 1, int(self.kernel_size / 8), 0, 0)))
        self.feature.add_module('f_batchnorm4', nn.BatchNorm2d(self.F2, False))
        self.feature.add_module('f_ELU4', nn.ELU())
        self.feature.add_module('f_Pooling4', nn.AvgPool2d(kernel_size=(1, 8)))
        self.feature.add_module('f_dropout4', nn.Dropout(p=0.25))

    def forward(self, input_data):
        _feature_ = self.feature(input_data)
        return _feature_


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    a = scio.loadmat(os.path.join('config', 'transfer_62To64.mat'))['transform_matrix']
    transfer_matrix0 = torch.tensor(a).to(torch.float32)
    transfer_matrix1 = torch.eye(64, 64).to(torch.float32)
    net = EEG_Infinity(transfer_matrix0, transfer_matrix1).cuda()
    test = torch.rand(64, 1, 64, 384).cuda()
    class_output, domain_output = net(input_data=test, domain=1, alpha=1)
    print(class_output.size())
    print(domain_output.size())
