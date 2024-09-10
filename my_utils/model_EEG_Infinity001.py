import torch.nn as nn
import torch
from torch.autograd import Function
import scipy.io as scio
import os
import torch.nn.functional as F


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN_EEG(nn.Module):

    def __init__(self, transfer_matrix_source, transfer_matrix_target, FIR_order=17, FIR_n=1, k_source=1.67007736772,
                 dc_source=0.0, k_target=1, dc_target=0.20967039880915994):
        super(DANN_EEG, self).__init__()
        self.feature_map_size = 192
        self.num_classes = 2
        self.num_channels = transfer_matrix_source.size()[0]
        # Defined the source domain alignment heads
        self.alignment_head_source = Alignment_head(transfer_matrix=transfer_matrix_source, k=k_source, dc=dc_source,
                                                    FIR_order=FIR_order, FIR_n=FIR_n)
        self.alignment_head_target = Alignment_head(transfer_matrix=transfer_matrix_target, k=k_target, dc=dc_target,
                                                    FIR_order=FIR_order, FIR_n=FIR_n)
        # Defined the feature extractor
        self.feature = Feature_endocer_EEGNet()
        # Defined the feature classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(128, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, self.num_classes))
        # Defined the domain classifier
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
        if domain == 0:
            alignment_head_output = self.alignment_head_source(input_data)
        else:
            alignment_head_output = self.alignment_head_target(input_data)
        feature = self.feature(alignment_head_output)
        feature = feature.view(-1, self.feature_map_size)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output, alignment_head_output

    def get_spatial_loss(self):
        diff = self.alignment_head_source.channel_transfer_matrix - self.alignment_head_target.channel_transfer_matrix
        return torch.norm(diff, p=2)

    def get_spatial_loss_angle(self):
        loss = 0
        for i in range(self.num_channels):
            v0 = self.alignment_head_source.channel_transfer_matrix[i, :]
            v1 = self.alignment_head_target.channel_transfer_matrix[i, :]
            loss += torch.sum(v0 * v1) / (torch.norm(v0) * torch.norm(v1))
        loss = loss / self.num_channels
        return loss


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
        # Create a 2D convolutional layer to implement the FIR filter
        self.FIR_order = FIR_order
        self.conv = nn.Conv2d(1, FIR_n, (1, FIR_order), padding=0, bias=False)

        # Initialize parameters to avoid extreme filtering effects
        self.initialize_parameters()

    def forward(self, x):
        # Add zero padding
        x_padded = F.pad(x, (int(self.conv.kernel_size[1] / 2), int(self.conv.kernel_size[1] / 2), 0, 0), mode='constant', value=0)
        # Apply convolution (using normalized weights)
        return F.conv2d(x_padded, self.conv.weight, padding=0)

    def initialize_parameters(self):
        """ Initialize the filter parameters so that each parameter value is 1/FIR_order """
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / self.FIR_order)
class Alignment_head(nn.Module):
    def __init__(self, transfer_matrix, k=1.0, dc=0.0, FIR_order=17, FIR_n=1):
        super(Alignment_head, self).__init__()
        self.k = k
        self.dc = dc
        self.channel_transfer_matrix = nn.Parameter(torch.eye(transfer_matrix.size()[0]))
        self.channel_transfer_matrix_fixed = transfer_matrix.cuda()
        self.channel_norm = ChannelNorm()
        self.domain_filter = FIR_convolution(FIR_n, FIR_order)

    def forward(self, input_data):
        input_data = input_data.to(torch.float32)
        input_data = self.channel_norm(input_data)
        input_data = torch.matmul(self.channel_transfer_matrix_fixed, input_data)
        input_data = torch.matmul(self.channel_transfer_matrix, input_data)
        input_data = self.domain_filter(input_data)
        return input_data

    def custom_zero_grad(self):
        self.channel_transfer_matrix.grad.data.zero_()
        self.domain_filter.conv.weight.grad.data.zero_()


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
        input_data = input_data.to(torch.float32)
        feature = self.feature(input_data)
        return feature


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    a = scio.loadmat(os.path.join('config', 'transfer_62To64.mat'))['transform_matrix']
    transfer_matrix0 = torch.tensor(a).to(torch.float32)
    transfer_matrix1 = torch.eye(64, 64).to(torch.float32)
    net = DANN_EEG(transfer_matrix0, transfer_matrix1).cuda()
    test = torch.rand(64, 1, 64, 384).cuda()
    class_output, domain_output = net(input_data=test, domain=1, alpha=1)
    print(class_output.size())
    print(domain_output.size())
