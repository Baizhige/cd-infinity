import torch.utils.data as data
import numpy as np
import os
import torch
# from .my_tool import channel_norm, cov_loss

def cov_loss(tensorA, tensorB):
    """
        Calculate the L2 distance between the average covariance matrices of two tensors (tensorA and tensorB).

        tensorA and tensorB are tensors with the shape (batchsize, 1, c, T).
        Each tensor contains batchsize samples, and each sample is a matrix with c rows and T columns.
    """

    def compute_mean_covariance(input_tensor):
        # Remove dimensions of size 1, changing the tensor shape to (batchsize, c, T)
        input_tensor = input_tensor.squeeze(1)
        norms = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        input_tensor_normalized = input_tensor / norms
        covariance_matrices = torch.matmul(input_tensor_normalized, input_tensor_normalized.transpose(1, 2))
        # Calculate the average covariance matrix

        mean_covariance = covariance_matrices.mean(dim=0)
        return mean_covariance.squeeze()

    # Calculate the average covariance matrix of the two tensors
    mean_covariance_A = compute_mean_covariance(tensorA)
    mean_covariance_B = compute_mean_covariance(tensorB)

    # Calculate the L2 distance as the loss
    loss = torch.norm(mean_covariance_A - mean_covariance_B, p=2)
    return loss

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
class EEGDataSet(data.Dataset):
    def __init__(self, data_root, data_list,
                 start=0,
                 datalen=256,
                 num_channel=64,
                 transforms=None,
                 isChannelNorm=0):
        self.root = data_root
        self.isChannelNorm = isChannelNorm
        self.start = start
        self.datalen = datalen
        self.num_channel = num_channel
        self.transforms = transforms
        self.eeg_data = np.load(os.path.join(data_root, data_list[0]))
        self.eeg_labels = np.load(os.path.join(data_root, data_list[1]))
        self.n_data = self.eeg_labels.shape[0]

    def __getitem__(self, item):
        eegs = self.eeg_data[item, :, :]
        labels = self.eeg_labels[item]
        subjects = 1
        eegs = eegs[:, self.start:self.start + self.datalen]
        eegs = np.expand_dims(eegs, axis=0)
        eegs = torch.tensor(eegs)
        labels = int(labels)
        return eegs, subjects, labels

    def __len__(self):
        return self.n_data


if __name__ == "__main__":
    import configparser
    import argparse

    parser = argparse.ArgumentParser(description='Read configuration file.')
    parser.add_argument('--config', default='config_PhysioNetMIToMengExp3.ini', help='Path to the config.ini file')
    parser.add_argument('--cache_prefix', default='parser_test', help='prefix of the cache (IMPORTANT!)')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join("..","hyperparameters", args.config))
    cross_id = 0
    source_eeg_root = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, "EEGData")
    source_train_list = [os.path.join(config.get('settings', 'source_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'source_datafile_name')),
                         os.path.join(config.get('settings', 'source_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'source_labelfile_name'))]

    target_eeg_root = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, "EEGData")

    target_train_list = [os.path.join(config.get('settings', 'target_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'target_datafile_name')),
                         os.path.join(config.get('settings', 'target_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'target_labelfile_name'))]


    source_train_dataset = EEGDataSet(
        data_root=source_eeg_root,
        data_list=source_train_list,
        num_channel=config.getint('settings', 'source_num_channel'),
        datalen=config.getint('settings', 'source_datalen')
    )
    source_train_dataloader = torch.utils.data.DataLoader(
        dataset=source_train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4)

    target_train_dataset = EEGDataSet(
        data_root=target_eeg_root,
        data_list=target_train_list,
        num_channel=config.getint('settings', 'target_num_channel'),
        datalen=config.getint('settings', 'target_datalen')
    )

    target_train_dataloader = torch.utils.data.DataLoader(
        dataset=target_train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4)

    source_train_iter = iter(source_train_dataloader)
    target_train_iter = iter(target_train_dataloader)
    batch1_sample_source,_,_ = next(source_train_iter)
    batch2_sample_source,_,_ = next(source_train_iter)

    batch1_sample_target,_,_ = next(target_train_iter)
    batch2_sample_target,_,_ = next(target_train_iter)



    def compute_mean_covariance(input_tensor):
        input_tensor = input_tensor.squeeze(1)
        norms = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        input_tensor_normalized = input_tensor / norms
        covariance_matrices = torch.matmul(input_tensor_normalized, input_tensor_normalized.transpose(1, 2))
        mean_covariance = covariance_matrices.mean(dim=0)
        return mean_covariance.squeeze()


    loss_ss = cov_loss(batch1_sample_source, batch2_sample_source)  # source-source loss
    loss_tt = cov_loss(batch1_sample_target, batch2_sample_target)  # target-target loss
    loss_st1 = cov_loss(batch1_sample_source, batch1_sample_target)  # source-target loss for batch 1
    loss_st2 = cov_loss(batch2_sample_source, batch2_sample_target)  # source-target loss for batch 2

    # Print the loss
    print(f"Source-Source Loss: {loss_ss.item():.4f}")
    print(f"Target-Target Loss: {loss_tt.item():.4f}")
    print(f"Source-Target Loss (Batch 1): {loss_st1.item():.4f}")
    print(f"Source-Target Loss (Batch 2): {loss_st2.item():.4f}")
