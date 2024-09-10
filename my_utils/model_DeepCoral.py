import torch
import torch.nn as nn
import torch.utils.data


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # After applying the pow() method to square each individual data point, sum along axis=1 (horizontally, from the first column to the last column). At this point, the shape of xx is (m, 1). After using the expand() method, it is expanded n-1 times, and the shape of xx becomes (m, n).
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy will be transposed at the end
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None), this line represents the operation dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # The clamp() function can restrict the elements in dist to a specified minimum and maximum range. Finally, dist is square-rooted to obtain the distance matrix between samples.
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist = torch.sum(dist)  # Sum all the distances
    return dist


class depthwise_separable_conv(nn.Module):  # Depthwise separable convolution
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DeepCoral(nn.Module):  # Net4: DeepCoral
    def __init__(self, transfer_matrix):
        super(DeepCoral, self).__init__()
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

        self.classifier_1 = nn.Sequential()
        self.classifier_1.add_module('fc-1', nn.Linear(self.feature_map_size, 128))
        self.classifier_1.add_module('fb-1', nn.BatchNorm1d(128))

        self.classifier_2 = nn.Sequential()
        self.classifier_2.add_module('fc-2', nn.Linear(128, 64))
        self.classifier_2.add_module('fb-2', nn.BatchNorm1d(64))

        self.classifier_3 = nn.Sequential()
        self.classifier_3.add_module('fc-3', nn.Linear(64, self.num_classes))

    def ori_forward(self, source_data, target_data):

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

    def forward(self, input_data, domain=None, alpha=None):
        if domain == 1:
            # target domain transfer
            input_data = torch.matmul(self.transfer_matrix, input_data)
        input_data = input_data.to(torch.float32)
        feature = self.feature(input_data)

        feature_1 = feature.view(-1, self.feature_map_size)
        feature_2 = self.classifier_1(feature_1)
        feature_3 = self.classifier_2(feature_2)
        output = self.classifier_3(feature_3)

        return output, feature_1, feature_2, feature_3
