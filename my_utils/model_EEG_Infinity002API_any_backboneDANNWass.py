import torch.nn as nn
import torch
from torch.autograd import Function
import scipy.io as scio
import os
import torch.nn.functional as F
from torch.nn.functional import elu
from collections import OrderedDict
from .model_utils.model_standard_deep4_util import np_to_th
from .model_utils.model_standard_deep4_modules import Expression, Ensure4d, AvgPool2dWithConv
from .model_utils.model_standard_deep4_functions import (
    identity, safe_log, square, transpose_time_to_spat, squeeze_final_output
)
from torch.nn import init

class EEG_Infinity(nn.Module):

    def __init__(self, transfer_matrix_source, transfer_matrix_target, num_channels = 64, FIR_order=17, FIR_n=1, backbone_type='InceptionEEG',right_idx=None, left_idx=None):
        super(EEG_Infinity, self).__init__()

        self.num_classes = 2
        self.num_channels = transfer_matrix_source.size()[0]

        # define alignment heads for source domain
        self.alignment_head_source = Alignment_head(transfer_matrix=transfer_matrix_source,
                                                    FIR_order=FIR_order, FIR_n=FIR_n)
        self.alignment_head_target = Alignment_head(transfer_matrix=transfer_matrix_target,
                                                    FIR_order=FIR_order, FIR_n=FIR_n)

        # define feature extractor
        if backbone_type == 'EEGNet':
            self.feature = EEGNetFeatureExtractor(num_channels=num_channels)
            self.feature_map_size = 192
        elif backbone_type == 'ShallowConvNet':
            self.feature = ShallowNetFeatureExtractor(num_channels=num_channels)
            self.feature_map_size = 800
        elif backbone_type == 'DeepConvNet':
            self.feature = DeepNetFeatureExtractor(num_channels=num_channels)
            self.feature_map_size = 600
        elif backbone_type == 'InceptionEEG':
            self.feature = InceptionEEGFeatureExtractor(num_channels=num_channels)
            self.feature_map_size = self.feature.__hidden_len__
        elif backbone_type == 'EEGSym':
            self.feature = EEGSymFeatureExtractor(right_idx=right_idx, left_idx=left_idx)
            self.feature_map_size = self.feature.feature_map_size
        else:
            raise ("error type of backbone")

        # define feature cls
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.feature_map_size, 128))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(128, 64))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(64, self.num_classes))
        # define domain cls
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc1', nn.Linear(self.feature_map_size, 128))
        self.domain_classifier.add_module('c_bn1', nn.BatchNorm1d(128))
        self.domain_classifier.add_module('c_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('c_drop1', nn.Dropout())
        self.domain_classifier.add_module('c_fc2', nn.Linear(128, 64))
        self.domain_classifier.add_module('c_bn2', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('c_relu2', nn.ReLU(True))
        # only one output
        self.domain_classifier.add_module('c_fc3', nn.Linear(64, 1))

    def clip_gradients_domain_classifier(self, threshold=0.01):
        """
        clip all parameter gradients of self.domain_classifier.

        :param threshold: truncation threshold, default is 0.01
        """
        torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), threshold)

    def forward(self, input_data, domain, alpha):
        input_data = input_data.to(torch.float32)
        if domain == 0:
            filter_output, _ = self.alignment_head_source(input_data)
        else:
            filter_output, _ = self.alignment_head_target(input_data)

        _feature_ = self.feature(filter_output).view(-1, self.feature_map_size)

        _reverse_feature_ = ReverseLayerF.apply(_feature_, alpha)

        class_output = self.class_classifier(_feature_)
        domain_output = self.domain_classifier(_reverse_feature_)

        return class_output, domain_output, None, None


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
        # Create a 2D convolution layer to implement a FIR filter
        self.FIR_order = FIR_order
        self.conv = nn.Conv2d(1, FIR_n, (1, FIR_order), padding=0, bias=False)

        # Initialize parameters to avoid extreme filtering effects
        self.initialize_parameters()

    def forward(self, x):
        # zero padding
        x_padded = F.pad(x, (int(self.conv.kernel_size[1] / 2), int(self.conv.kernel_size[1] / 2), 0, 0),
                         mode='constant', value=0)
        # Apply convolution (with normalized weights)
        return F.conv2d(x_padded, self.conv.weight, padding=0)

    def initialize_parameters(self):
        """ Initialize the filter parameters so that the value of each parameter is 1/FIR_order """
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / self.FIR_order)


class Alignment_head(nn.Module):
    def __init__(self, transfer_matrix, FIR_order=17, FIR_n=1):
        super(Alignment_head, self).__init__()
        self.channel_transfer_matrix_fixed = transfer_matrix.cuda()

    def forward(self, input_data):
        _input_data_ = input_data.to(torch.float32)
        _input_data_ = torch.matmul(self.channel_transfer_matrix_fixed, _input_data_)
        return _input_data_, None



class EEGNetFeatureExtractor(nn.Module):
    def __init__(self, kernel_size=64, num_channels=64):
        super(EEGNetFeatureExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = num_channels
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


class ShallowNetFeatureExtractor(nn.Module):
    def __init__(self,
                 num_channels = 64,
                 input_window_samples=None,
                 n_filters_time=40,
                 filter_time_length=25,
                 n_filters_spat=40,
                 pool_time_length=75,
                 pool_time_stride=15,
                 final_conv_length=30,
                 conv_nonlin=square,
                 pool_mode="mean",
                 pool_nonlin=safe_log,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob=0.5):
        super(ShallowNetFeatureExtractor, self).__init__()
        self.in_chans = num_channels
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob

        self._build_network()
        self._initialize_weights()

    def _build_network(self):
        self.ensuredims = Ensure4d()
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        if self.split_first_layer:
            self.conv_time = nn.Conv2d(
                1,
                self.n_filters_time,
                (self.filter_time_length, 1),
                stride=1,
            )
            self.conv_spat = nn.Conv2d(
                self.n_filters_time,
                self.n_filters_spat,
                (1, self.in_chans),
                stride=1,
                bias=not self.batch_norm,
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.conv_time = nn.Conv2d(
                self.in_chans,
                self.n_filters_time,
                (self.filter_time_length, 1),
                stride=1,
                bias=not self.batch_norm,
            )
            n_filters_conv = self.n_filters_time

        if self.batch_norm:
            self.bnorm = nn.BatchNorm2d(
                n_filters_conv, momentum=self.batch_norm_alpha, affine=True
            )

        self.conv_nonlin_exp = Expression(self.conv_nonlin)

        self.pool = pool_class(
            kernel_size=(self.pool_time_length, 1),
            stride=(self.pool_time_stride, 1),
        )

        self.pool_nonlin_exp = Expression(self.pool_nonlin)

        self.drop = nn.Dropout(p=self.drop_prob)

        feature_layers = [
            self.ensuredims,
            Expression(transpose_time_to_spat) if self.split_first_layer else nn.Identity(),
            self.conv_time,
            self.conv_spat if self.split_first_layer else nn.Identity(),
            self.bnorm if self.batch_norm else nn.Identity(),
            self.conv_nonlin_exp,
            self.pool,
            self.pool_nonlin_exp,
            self.drop
        ]

        self.feature_extractor = nn.Sequential(*feature_layers)

    def _initialize_weights(self):
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        if self.split_first_layer or not self.batch_norm:
            if self.conv_time.bias is not None:
                init.zeros_(self.conv_time.bias)

        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.zeros_(self.conv_spat.bias)

        if self.batch_norm:
            init.ones_(self.bnorm.weight)
            init.zeros_(self.bnorm.bias)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = x.permute(0, 2, 3, 1)
        features = self.feature_extractor(x)
        return features


class DeepNetFeatureExtractor(nn.Sequential):
    def __init__(
            self,
            num_channels=64,
            num_class=2,
            len_window=384,
            final_conv_length='auto',
            n_filters_time=25,
            n_filters_spat=25,
            filter_time_length=10,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=50,
            filter_length_2=3,
            n_filters_3=100,
            filter_length_3=3,
            n_filters_4=200,
            filter_length_4=3,
            first_conv_nonlin=elu,
            first_pool_mode="max",
            first_pool_nonlin=identity,
            later_conv_nonlin=elu,
            later_pool_mode="max",
            later_pool_nonlin=identity,
            drop_prob=0.5,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False
    ):
        super().__init__()
        self.in_chans = num_channels
        self.n_classes = num_class
        self.input_window_samples = len_window
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_conv_nonlin = first_conv_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_conv_nonlin = later_conv_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        self._build_feature_extractor()
        self._initialize_weights()

    def _build_feature_extractor(self):
        # Feature extractor construction
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]

        feature_layers = []

        feature_layers.append(("ensuredims", Ensure4d()))
        feature_layers.append(("dimshuffle", Expression(transpose_time_to_spat)))

        if self.split_first_layer:
            feature_layers.append(
                ("conv_time",
                 nn.Conv2d(
                     1, self.n_filters_time, (self.filter_time_length, 1), stride=1
                 ))
            )
            feature_layers.append(
                ("conv_spat",
                 nn.Conv2d(
                     self.n_filters_time, self.n_filters_spat,
                     (1, self.in_chans), stride=(self._get_conv_stride(), 1),
                     bias=not self.batch_norm
                 ))
            )
            n_filters_conv = self.n_filters_spat
        else:
            feature_layers.append(
                ("conv_time",
                 nn.Conv2d(
                     self.in_chans, self.n_filters_time,
                     (self.filter_time_length, 1), stride=(self._get_conv_stride(), 1),
                     bias=not self.batch_norm
                 ))
            )
            n_filters_conv = self.n_filters_time

        if self.batch_norm:
            feature_layers.append(
                ("bnorm",
                 nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha, affine=True, eps=1e-5))
            )

        feature_layers.append(("conv_nonlin", Expression(self.first_conv_nonlin)))
        feature_layers.append(
            ("pool",
             first_pool_class(kernel_size=(self.pool_time_length, 1), stride=(self._get_pool_stride(), 1)))
        )
        feature_layers.append(("pool_nonlin", Expression(self.first_pool_nonlin)))

        self._add_conv_pool_block(feature_layers, n_filters_conv, self.n_filters_2, self.filter_length_2, 2)
        self._add_conv_pool_block(feature_layers, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3)
        self._add_conv_pool_block(feature_layers, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4)

        self.feature_extractor = nn.Sequential(OrderedDict(feature_layers))

    def _initialize_weights(self):
        # Initialize weights for feature extractor
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
    def _get_conv_stride(self):
        return self.pool_time_stride if self.stride_before_pool else 1

    def _get_pool_stride(self):
        return 1 if self.stride_before_pool else self.pool_time_stride

    def _add_conv_pool_block(self, feature_layers, n_filters_before, n_filters, filter_length, block_nr):
        suffix = f"_{block_nr}"
        feature_layers.append(
            ("drop" + suffix, nn.Dropout(p=self.drop_prob))
        )
        feature_layers.append(
            ("conv" + suffix,
             nn.Conv2d(
                 n_filters_before, n_filters, (filter_length, 1), stride=(self._get_conv_stride(), 1),
                 bias=not self.batch_norm
             ))
        )
        if self.batch_norm:
            feature_layers.append(
                ("bnorm" + suffix,
                 nn.BatchNorm2d(n_filters, momentum=self.batch_norm_alpha, affine=True, eps=1e-5))
            )
        feature_layers.append(("nonlin" + suffix, Expression(self.later_conv_nonlin)))

        pool_class = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)[self.later_pool_mode]
        feature_layers.append(
            ("pool" + suffix,
             pool_class(kernel_size=(self.pool_time_length, 1), stride=(self._get_pool_stride(), 1)))
        )
        feature_layers.append(("pool_nonlin" + suffix, Expression(self.later_pool_nonlin)))

    def forward(self, input_data):
        # Define the forward pass
        input_data = input_data.type(torch.cuda.FloatTensor)
        input_data = input_data.permute(0, 2, 3, 1)
        features = self.feature_extractor(input_data)
        # print(features.shape)
        return features
class InceptionEEGNet_Block1(nn.Module):
    def __init__(self, kernel_size, num_channel=64):
        super(InceptionEEGNet_Block1, self).__init__()
        self.F = 8
        self.D = 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, kernel_size)),
            nn.ZeroPad2d((int(kernel_size / 2) - 1, int(kernel_size / 2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F * self.D, kernel_size=(num_channel, 1), groups=self.F),
            nn.BatchNorm2d(self.F * self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, int(kernel_size / 2))),
            nn.ZeroPad2d((int(kernel_size / 4) - 1, int(kernel_size / 4), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F * self.D, kernel_size=(num_channel, 1), groups=self.F),
            nn.BatchNorm2d(self.F * self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, self.F, kernel_size=(1, int(kernel_size / 4))),
            nn.ZeroPad2d((int(kernel_size / 8) - 1, int(kernel_size / 8), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(self.F, self.F * self.D, kernel_size=(num_channel, 1), groups=self.F),
            nn.BatchNorm2d(self.F * self.D, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch_pool = nn.AvgPool2d(kernel_size=(1, 4))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        N1 = torch.cat((branch1, branch2, branch3), dim=1)
        A1 = self.branch_pool(N1)
        return A1


class InceptionEEGNet_Block2(nn.Module):
    def __init__(self, kernel_size, num_channel=64):
        super(InceptionEEGNet_Block2, self).__init__()
        self.F = 8
        self.D = 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size / 4))),
            nn.ZeroPad2d((int(kernel_size / 8) - 1, int(kernel_size / 8), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size / 8))),
            nn.ZeroPad2d((int(int(kernel_size / 8) / 2) - 1, int(int(kernel_size / 8) / 2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(48, self.F, kernel_size=(1, int(kernel_size / 16))),
            nn.ZeroPad2d((int(int(kernel_size / 16) / 2), int(int(kernel_size / 16) / 2), 0, 0)),
            nn.BatchNorm2d(self.F, False),
            nn.ELU(),
            nn.Dropout(p=0.25)
        )
        self.branch_pool = nn.AvgPool2d(kernel_size=(1, 2))

    def forward(self, x):
        branch1 = self.branch1(x)
        # print(branch1.size())
        branch2 = self.branch2(x)
        # print(branch2.size())
        branch3 = self.branch3(x)
        # print(branch3.size())
        N2 = torch.cat((branch1, branch2, branch3), dim=1)
        A2 = self.branch_pool(N2)
        return A2


class InceptionEEGFeatureExtractor(nn.Module):

    def __init__(self, num_channels = 64, num_class = 2, len_window = 384):
        super(InceptionEEGFeatureExtractor, self).__init__()
        # feature extractor
        self.kernel_size = 80
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.num_channel = num_channels
        self.n_classes = num_class
        self.len_window = len_window
        self.feature = nn.Sequential()
        # (N,1,64,256)
        self.feature.add_module('f_block1', InceptionEEGNet_Block1(kernel_size=80, num_channel=self.num_channel))
        # (N,48,1,256/4)
        self.feature.add_module('f_block2', InceptionEEGNet_Block2(kernel_size=80, num_channel=self.num_channel))
        # (N,24,1,256/4/2)
        self.feature.add_module('f_conv3', nn.Conv2d(24, 12, kernel_size=(1, int(self.kernel_size / 8))))
        self.feature.add_module('f_padding3',
                                nn.ZeroPad2d((int(self.kernel_size / 16) - 1, int(self.kernel_size / 16), 0, 0)))
        self.feature.add_module('f_batchnorm3', nn.BatchNorm2d(12, False))
        self.feature.add_module('f_ELU3', nn.ELU())

        self.feature.add_module('f_dropout3', nn.Dropout(p=0.25))
        self.feature.add_module('f_pooling3',
                                nn.AvgPool2d(kernel_size=(1, 2)))
        # (N,12,1,256/4/2/2)
        self.feature.add_module('f_conv4', nn.Conv2d(12, 6, kernel_size=(1, int(self.kernel_size / 16))))
        self.feature.add_module('f_padding4',
                                nn.ZeroPad2d((int(self.kernel_size / 32), int(self.kernel_size / 32), 0, 0)))
        self.feature.add_module('f_batchnorm4', nn.BatchNorm2d(6, False))
        self.feature.add_module('f_ELU4', nn.ELU())
        self.feature.add_module('f_dropout4', nn.Dropout(p=0.25))
        self.feature.add_module('f_pooling4',
                                nn.AvgPool2d(kernel_size=(1, 2)))
        __hidden_feature__ = self.feature(torch.rand(1, 1, self.num_channel, self.len_window))
        self.__hidden_len__ = __hidden_feature__.shape[1] * __hidden_feature__.shape[2] * __hidden_feature__.shape[3]

    def forward(self, input_data):
        input_data = input_data.type(torch.cuda.FloatTensor)
        feature = self.feature(input_data)
        return feature

class Symmetric_layer(nn.Module):
    def __init__(self, right_idx, left_idx):
        super(Symmetric_layer, self).__init__()
        self.right_idx = right_idx
        self.left_idx = left_idx

    def forward(self, x):
        '''
        :param x: B N C T
        :return: B N C T 2
        '''
        return torch.cat((x[:, :, self.right_idx, :].unsqueeze(4),
                      x[:, :, self.left_idx, :].unsqueeze(4)), dim=4)


class EEGSym_inception_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, kernel_size, padding, dropoutRate=0.3):
        super(EEGSym_inception_block, self).__init__()
        self.inChannel = inChannel
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel

        self.t_conv1 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[0], 1), stride=(1, 1, 1),
                      padding=(0, padding[0], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )


        self.t_conv2 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[1], 1), stride=(1, 1, 1),
                      padding=(0, padding[1], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.t_conv3 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[2], 1), stride=(1, 1, 1),
                      padding=(0, padding[2], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.pooling = nn.AvgPool3d(kernel_size=(1, 2, 1))

        self.group_conv = nn.Sequential(
            nn.Conv3d(hiddenChannel * 3, outChannel, (kernel_size[3], 1, 1), stride=(1, 1, 1),
                      padding=(padding[3], 0, 0), groups=hiddenChannel * 3),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.conv_res1 = nn.Conv3d(inChannel, hiddenChannel * 3, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn_res1 = nn.BatchNorm3d(hiddenChannel * 3)

        if hiddenChannel * 3 != outChannel:
            self.conv_res2 = nn.Conv3d(hiddenChannel * 3, outChannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
            self.bn_res2 = nn.BatchNorm3d(outChannel)

    def forward(self, x):
        brand1 = self.t_conv1(x)
        brand2 = self.t_conv2(x)
        brand3 = self.t_conv3(x)
        res1 = self.bn_res1(self.conv_res1(x))
        out_1 = torch.add(torch.cat((brand1, brand2, brand3), dim=1), res1)  # torch.add会自动广播
        out_2 = self.pooling(out_1)
        out_3 = self.group_conv(out_2)
        if self.hiddenChannel * 3 != self.outChannel:
            res2 = self.bn_res2(self.conv_res2(out_2))
        else:
            res2 = out_2
        out = torch.add(out_3, res2)
        return out


class EEGSym_residual_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, kernel_size, padding, dropoutRate=0.3):
        super(EEGSym_residual_block, self).__init__()
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel

        self.t_conv1 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[0], 1), stride=(1, 1, 1),
                      padding=(0, padding[0], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )

        self.t_conv2 = nn.Sequential(
            nn.Conv3d(inChannel, hiddenChannel, (1, kernel_size[1], 1), stride=(1, 1, 1),
                      padding=(0, padding[1], 0)),
            nn.BatchNorm3d(hiddenChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )
        self.t_conv3 = nn.Sequential(
            nn.Conv3d(hiddenChannel, outChannel, (kernel_size[2], 1, 1), stride=(1, 1, 1),
                      padding=(padding[2], 0, 0)),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )



        self.pooling = nn.AvgPool3d(kernel_size=(1, 2, 1))
        if hiddenChannel != outChannel:
            self.conv_res = nn.Conv3d(hiddenChannel, outChannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
            self.bn_res = nn.BatchNorm3d(outChannel)

    def forward(self, x):
        out_1 = self.t_conv1(x) + self.t_conv2(x)
        out_2 = self.pooling(out_1)
        out_3 = self.t_conv3(out_2)
        if self.hiddenChannel != self.outChannel:
            res1 = self.bn_res(self.conv_res(out_2))
        else:
            res1 = out_2
        return torch.add(out_3, res1)


class EEGSym_residual_mini_block(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size, padding, dropoutRate=0.3):
        super(EEGSym_residual_mini_block, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.t_conv1 = nn.Sequential(
            nn.Conv3d(inChannel, outChannel, (1, kernel_size[0], 1), stride=(1, 1, 1),
                      padding=(0, padding[0], 0)),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )

        if inChannel != outChannel:
            self.conv_res = nn.Conv3d(inChannel, outChannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
            self.bn_res = nn.BatchNorm3d(outChannel)

    def forward(self, x):
        if self.inChannel != self.outChannel:
            res = self.bn_res(self.conv_res(x))
        else:
            res = x
        out = self.t_conv1(x)
        return torch.add(out, res)


class EEGSym_Channel_Merging_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, numChannel, dropoutRate=0.3):
        super(EEGSym_Channel_Merging_block, self).__init__()
        self.inChannel = inChannel
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel
        self.num_channel = numChannel
        self.res_block1 = EEGSym_residual_mini_block(inChannel, hiddenChannel, [5], [2])
        self.res_block2 = EEGSym_residual_mini_block(inChannel, hiddenChannel, [5], [2])
        self.channel_merging_block = nn.Sequential(
            nn.Conv3d(hiddenChannel, outChannel, (numChannel, 1, 2), stride=(1, 1, 1),
                      padding=(0, 0, 0), groups=9),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )



    def forward(self, x):
        output = self.res_block1(x)
        output = self.res_block2(output)
        output = self.channel_merging_block(output)
        return output


class EEGSym_Temporal_Merging_block(nn.Module):
    def __init__(self, inChannel, hiddenChannel, outChannel, NumTemperal, dropoutRate=0.3):
        super(EEGSym_Temporal_Merging_block, self).__init__()
        self.inChannel = inChannel
        self.hiddenChannel = hiddenChannel
        self.outChannel = outChannel
        self.NumTemperal = NumTemperal
        self.res_block1 = EEGSym_residual_mini_block(inChannel, hiddenChannel, [NumTemperal], [0])
        self.g_conv1 = nn.Sequential(
            nn.Conv3d(hiddenChannel, outChannel, (1, NumTemperal, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0), groups=hiddenChannel),
            nn.BatchNorm3d(outChannel),
            nn.ELU(),
            nn.Dropout(p=dropoutRate)
        )


    def forward(self, x):
        output = self.res_block1(x)
        output = self.g_conv1(output)
        return output




class EEGSymFeatureExtractor(nn.Module):
    def __init__(self, right_idx=None, left_idx=None, feature_map_size=None, num_classes=None):
        '''
        :param right_idx: channel index for electrodes located in right hemisphere
        :param left_idx:  channel index for electrodes located in left hemisphere
        '''
        super(EEGSymFeatureExtractor, self).__init__()
        # Parameters----------------------------------------------
        if feature_map_size == None:
            self.feature_map_size = 36
        else:
            self.feature_map_size = feature_map_size
        if num_classes == None:
            self.num_classes = 4
        else:
            self.num_classes = num_classes
        if right_idx is None or left_idx is None:
            self.sym_layer = Symmetric_layer([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
            self.num_channels = 5
        else:
            self.sym_layer = Symmetric_layer(right_idx, left_idx)
            self.num_channels=len(right_idx)
        # Convolution Block----------------------------------------------
        self.Block1 = nn.Sequential(
            EEGSym_inception_block(1, 24, 72, [64+1, 32+1, 16+1, self.num_channels], [32, 16, 8, 0]),
            EEGSym_inception_block(72, 24, 72, [16+1, 8+1, 4+1, self.num_channels], [8, 4, 2, 0])
        )
        self.Block2 = nn.Sequential(
            EEGSym_residual_block(72, 36, 36, [1, 16+1, self.num_channels], [0, 8, 0]),
            EEGSym_residual_block(36, 36, 36, [1, 8+1, self.num_channels], [0, 4, 0]),
            EEGSym_residual_block(36, 18, 18, [1, 4+1, self.num_channels], [0, 2, 0])
        )
        self.Block3 = nn.Sequential(
            EEGSym_residual_mini_block(18, 18, [5], [2]),
            nn.AvgPool3d(kernel_size=(1, 2, 1))
        )
        self.Block4 = EEGSym_Channel_Merging_block(18, 18, 18, self.num_channels).cuda()
        self.Block5 = EEGSym_Temporal_Merging_block(18, 18, 36, 6).cuda()
        self.Block6 = nn.Sequential(
            EEGSym_residual_mini_block(36, 36, [1], [0]),
            EEGSym_residual_mini_block(36, 36, [1], [0]),
            EEGSym_residual_mini_block(36, 36, [1], [0]),
            EEGSym_residual_mini_block(36, 36, [1], [0]),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feature_map_size, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.25),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.25),
            nn.Linear(16, self.num_classes)
        )

    def forward(self, input_data):
        input_data = input_data.type(torch.cuda.FloatTensor)
        torch.cuda.empty_cache()
        output = self.sym_layer(input_data)
        torch.cuda.empty_cache()
        output = self.Block1(output)
        torch.cuda.empty_cache()
        output = self.Block2(output)
        torch.cuda.empty_cache()
        output = self.Block3(output)
        torch.cuda.empty_cache()
        output = self.Block4(output)
        torch.cuda.empty_cache()
        output = self.Block5(output)
        torch.cuda.empty_cache()
        output = self.Block6(output)
        torch.cuda.empty_cache()
        return output



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
