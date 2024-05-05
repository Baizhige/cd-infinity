import torch
import os
import argparse
import configparser
import torch.nn as nn
from my_utils.model_EEG_Infinity003Wass_any_backbone import FIR_convolution
def get_Para(method, cross_id, task):
    # 定义模型文件名和文件夹路径
    model_name = f'Comparison_{method}_InceptionEEG_{task}_1_cross_id_{cross_id}_best_target_model.pth'
    models_folder = 'models'

    parser = argparse.ArgumentParser(description='Read configuration file.')
    parser.add_argument('--config', default=f'config_{task}.ini', help='Path to the config.ini file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join("hyperparameters", args.config))

    # 使用torch.load加载模型
    model_path = os.path.join(models_folder, model_name)

    model = torch.load(model_path, map_location=torch.device('cpu'))

    return model.alignment_head_source.channel_transfer_matrix.detach().numpy().squeeze(), model.alignment_head_source.domain_filter.conv.weight.detach().numpy().squeeze(), model.alignment_head_target.channel_transfer_matrix.detach().numpy().squeeze(), model.alignment_head_target.domain_filter.conv.weight.detach().numpy().squeeze()


import numpy as np
import matplotlib.pyplot as plt


def draw_model_parameter(matrix1, matrix2, vector1, vector2):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)

    # 设置颜色映射
    cmap = 'viridis'

    # 绘制前两个矩阵
    for i, matrix in enumerate([matrix1, matrix2]):
        cax = axes[i].matshow(matrix, cmap=cmap)
        fig.colorbar(cax, ax=axes[i])
        axes[i].set_title(f'Matrix {i + 1}')
        axes[i].tick_params(which='both', direction='in')
        axes[i].xaxis.set_ticks_position('bottom')

    # 绘制后两个向量的柱状图
    for i, vector in enumerate([vector1, vector2], start=2):
        axes[i].bar(np.arange(len(vector)), vector, color=plt.cm.viridis(np.linspace(0, 1, len(vector))))
        axes[i].set_title(f'Vector {i - 1}')
        axes[i].tick_params(which='both', direction='in')
        axes[i].minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join('figures','model_parameters.pdf'), format='pdf', dpi=300)

def get_data(cross_id, task):
    parser = argparse.ArgumentParser(description='Read configuration file.')
    parser.add_argument('--config', default=f'config_{task}.ini', help='Path to the config.ini file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join("hyperparameters", args.config))

    source_eeg_root = os.path.join(os.path.pardir, os.path.pardir, "EEGData")
    source_train_list = [os.path.join(config.get('settings', 'source_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'source_datafile_name')),
                         os.path.join(config.get('settings', 'source_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'source_labelfile_name'))]

    target_eeg_root = os.path.join(os.path.pardir, os.path.pardir, "EEGData")

    target_train_list = [os.path.join(config.get('settings', 'target_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'target_datafile_name')),
                         os.path.join(config.get('settings', 'target_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'target_labelfile_name'))]

    source_train_data = np.load(os.path.join(source_eeg_root, source_train_list[0]))
    target_train_data = np.load(os.path.join(target_eeg_root, target_train_list[0]))
    return source_train_data, target_train_data

def get_prior_transfer(task):
    config = configparser.ConfigParser()
    config.read(os.path.join("hyperparameters", f"config_{task}.ini"))
    CAR_matrix_source = torch.eye(config.getint('settings', 'source_num_channel')) - torch.ones(
        [config.getint('settings', 'source_num_channel'),
         config.getint('settings', 'source_num_channel')]) / config.getint('settings', 'source_num_channel')

    transfer_matrix_source = CAR_matrix_source

    CAR_matrix_target = torch.eye(config.getint('settings', 'target_num_channel')) - torch.ones(
        [config.getint('settings', 'target_num_channel'),
         config.getint('settings', 'target_num_channel')]) / config.getint('settings', 'target_num_channel')
    transfer_matrix_target = torch.matmul(
        torch.tensor(np.load(os.path.join('config', config.get('settings', 'file_name_transfer_matrix')))).to(
            torch.float32), CAR_matrix_target)

    return transfer_matrix_source, transfer_matrix_target


def transform_data(method, cross_id, task):
    # get prior transfer matrix
    transfer_matrix_source, transfer_matrix_target = get_prior_transfer(task)

    # get model parameters
    source_SFB, source_FFB, target_SFB, target_FFB = get_Para(method, cross_id, task=task)

    # get data
    source_train_data, target_train_data = get_data(cross_id, task)

    # transform data
    source_prior_transformed_data = np.einsum('ij,kjl->kil', transfer_matrix_source, source_train_data)
    target_prior_transformed_data = np.einsum('ij,kjl->kil', transfer_matrix_target, target_train_data)

    source_adjust_transformed_data = np.einsum('ij,kjl->kil', source_SFB, source_prior_transformed_data)
    target_adjust_transformed_data = np.einsum('ij,kjl->kil', target_SFB, target_prior_transformed_data)

    # ----
    source_domain_filter = FIR_convolution(1, 17)
    source_domain_filter.conv.weight = nn.Parameter(torch.from_numpy(source_FFB).float().view(1, 1, 1, 17))

    target_domain_filter = FIR_convolution(1, 17)
    target_domain_filter.conv.weight = nn.Parameter(torch.from_numpy(target_FFB).float().view(1, 1, 1, 17))

    source_input_tensor = torch.from_numpy(source_adjust_transformed_data).float().unsqueeze(1)  # 维度变为(batch_size, 1, height, width)
    target_input_tensor = torch.from_numpy(target_adjust_transformed_data).float().unsqueeze(1)  # 维度变为(batch_size, 1, height, width)
    # 使用torch.no_grad()禁用梯度计算
    with torch.no_grad():
        # 使用conv_layer进行卷积操作
        source_output_tensor = source_domain_filter(source_input_tensor)
        target_output_tensor = target_domain_filter(target_input_tensor)

    source_FIR_transformed_data = source_output_tensor.numpy().squeeze()
    target_FIR_transformed_data = target_output_tensor.numpy().squeeze()

    print(source_train_data.shape)
    print(target_train_data.shape)

    print(source_prior_transformed_data.shape)
    print(target_prior_transformed_data.shape)

    print(source_adjust_transformed_data.shape)
    print(target_adjust_transformed_data.shape)

    print(source_FIR_transformed_data.shape)
    print(target_FIR_transformed_data.shape)

    np.save(os.path.join("record", "source_train_data.npy"), source_train_data)
    np.save(os.path.join("record", "target_train_data.npy"), target_train_data)

    np.save(os.path.join("record", "source_prior_transformed_data.npy"), source_prior_transformed_data)
    np.save(os.path.join("record", "target_prior_transformed_data.npy"), target_prior_transformed_data)

    np.save(os.path.join("record", "source_adjust_transformed_data.npy"), source_adjust_transformed_data)
    np.save(os.path.join("record", "target_adjust_transformed_data.npy"), target_adjust_transformed_data)

    np.save(os.path.join("record", "source_FIR_transformed_data.npy"), source_FIR_transformed_data)
    np.save(os.path.join("record", "target_FIR_transformed_data.npy"), target_FIR_transformed_data)

    return None

# 定义模型和交叉验证ID
cross_id = 0
# source_SFB, source_FFB, target_SFB, target_FFB = get_Para("EEG_Infinity005Wass", cross_id, task="MengExp12ToMengExp3")
# draw_model_parameter(source_SFB, target_SFB, source_FFB, target_FFB)

transform_data("EEG_Infinity006Wass", cross_id, task="MengExp12ToMengExp3")