import numpy as np
import torch
from functools import reduce
from pyriemann.utils.mean import mean_covariance

class EuclideanMeanCovariance(object):
    def __init__(self, training_data, precision='float64', is_cuda=False):
        '''
        Initialize the EuclideanMeanCovariance class.

        Args:
            training_data (numpy.ndarray or torch.tensor): Training data with shape (num_samples, num_channels, num_sampling_points).
            precision (str): Data precision type, either 'float32' or 'float64'. Defaults to 'float64'.
            is_cuda (bool): Whether to use CUDA for computations. Defaults to False.

        Raises:
            ValueError: If the precision argument is not 'float32' or 'float64'.
            TypeError: If the training data is neither a numpy.ndarray nor a torch.tensor.
        '''
        if precision not in ['float32', 'float64']:
            raise ValueError("Precision must be 'float32' or 'float64'")

        if isinstance(training_data, torch.Tensor):
            training_data = training_data.numpy()
        elif not isinstance(training_data, np.ndarray):
            raise TypeError("training_data must be a numpy.ndarray or torch.tensor")

        training_data = training_data.astype(precision)

        R_source = np.zeros((training_data.shape[1], training_data.shape[1]), dtype=precision)
        R_source = reduce(lambda R, y: R + np.dot(y, y.T), training_data, R_source)
        R_source = R_source / training_data.shape[0]

        v, Q = np.linalg.eig(R_source)
        ss = np.diag(v ** (-0.5))
        ss[np.isnan(ss)] = 0

        self.re = torch.tensor(np.dot(Q, np.dot(ss, np.linalg.inv(Q))), dtype=getattr(torch, precision))
        self.num_channels = training_data.shape[1]

        if is_cuda:
            self.re = self.re.cuda()

    def transform(self, data):
        '''
        Transform the given data using the calculated matrix.

        Args:
            data (torch.tensor or numpy.ndarray): Data to be transformed.
                It can have shape:
                1. (num_channels, num_sampling_points),
                2. (batchsizes, num_channels, num_sampling_points) or
                3. (batchsizes, 1, num_channels, num_sampling_points)

        Returns:
            torch.tensor: Transformed data.
        '''
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=self.re.dtype)

        if self.re.is_cuda:
            data = data.cuda()
        if len(data.shape) == 2:
            return torch.matmul(self.re, data)
        elif len(data.shape) == 3:
            return torch.matmul(self.re.unsqueeze(0), data)
        elif len(data.shape) == 4:
            transformed_data = torch.matmul(self.re.unsqueeze(0), data.squeeze(1))
            return transformed_data.unsqueeze(1)
        else:
            raise ValueError("Unsupported data shape.")

class RiemannMeanCovariance(object):
    def __init__(self, training_data, precision='float64', is_cuda=False, metric='riemann'):
        '''
        Initialize the RiemannMeanCovariance class.

        Args:
            training_data (numpy.ndarray or torch.tensor): Training data with shape (num_samples, num_channels, num_sampling_points).
            precision (str): Data precision type, either 'float32' or 'float64'. Defaults to 'float64'.
            is_cuda (bool): Whether to use CUDA for computations. Defaults to False.
            metric (str): The metric for mean covariance. Defaults to 'riemann'.

        Raises:
            ValueError: If the precision argument is not 'float32' or 'float64'.
            TypeError: If the training data is neither a numpy.ndarray nor a torch.tensor.
        '''
        if precision not in ['float32', 'float64']:
            raise ValueError("Precision must be 'float32' or 'float64'")

        if isinstance(training_data, torch.Tensor):
            training_data = training_data.numpy()
        elif not isinstance(training_data, np.ndarray):
            raise TypeError("training_data must be a numpy.ndarray or torch.tensor")

        training_data = training_data.astype(precision)

        # Compute covariance matrices for each sample
        cov_matrices = np.array([np.cov(y) for y in training_data])

        # Compute the Riemannian mean of the covariance matrices
        self.riemann_mean = mean_covariance(cov_matrices, metric=metric)

        self.num_channels = training_data.shape[1]
        self.re = torch.tensor(self.riemann_mean, dtype=getattr(torch, precision))

        if is_cuda:
            self.re = self.re.cuda()

    def transform(self, data):
        '''
        Transform the given data using the calculated matrix.

        Args:
            data (torch.tensor or numpy.ndarray): Data to be transformed.
                It can have shape:
                1. (num_channels, num_sampling_points),
                2. (batchsizes, num_channels, num_sampling_points) or
                3. (batchsizes, 1, num_channels, num_sampling_points)

        Returns:
            torch.tensor: Transformed data.
        '''
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=self.re.dtype)

        if self.re.is_cuda:
            data = data.cuda()
        if len(data.shape) == 2:
            return torch.matmul(self.re, data)
        elif len(data.shape) == 3:
            return torch.matmul(self.re.unsqueeze(0), data)
        elif len(data.shape) == 4:
            transformed_data = torch.matmul(self.re.unsqueeze(0), data.squeeze(1))
            return transformed_data.unsqueeze(1)
        else:
            raise ValueError("Unsupported data shape.")

if __name__ == "__main__":
    import os
    training_data = np.load(os.path.join("..","..","..","EEGData","dataset_PhysioNetMI","concatedData","train","cross_4_dataInv_128_T64_None_default.npy"))
    print(training_data.shape)