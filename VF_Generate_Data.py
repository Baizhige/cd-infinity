import torch
import os
import argparse
import configparser
from my_utils.data_loader_npy import EEGDataSet

def process_method(method, cross_id, task):
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

    # Import Data
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

    source_train_dataset = EEGDataSet(
        data_root=source_eeg_root,
        data_list=source_train_list,
        num_channel=config.getint('settings', 'source_num_channel'),
        datalen=config.getint('settings', 'source_datalen')
    )
    source_train_dataloader = torch.utils.data.DataLoader(
        dataset=source_train_dataset,
        batch_size=config.getint('settings', 'batch_size'),
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
        batch_size=config.getint('settings', 'batch_size'),
        shuffle=True,
        num_workers=4)

    data_source_iter = iter(source_train_dataloader)
    data_target_iter = iter(target_train_dataloader)

    len_dataloader = min(len(source_train_dataloader), len(target_train_dataloader)) - 1

    s_FC_features = torch.zeros((len_dataloader * config.getint('settings', 'batch_size'), data_size))
    t_FC_features = torch.zeros((len_dataloader * config.getint('settings', 'batch_size'), data_size))

    for i in range(len_dataloader):
        print(i)
        # 正向传播源域数据
        data_source = next(data_source_iter)
        s_eeg, s_subject, s_label = data_source
        s_FC_feature = model.feature.feature(s_eeg).view(-1, data_size)
        s_FC_features[i * 32: (i + 1) * 32, :] = s_FC_feature.squeeze()  # 使用squeeze确保维度匹配

        data_target = next(data_target_iter)
        t_eeg, t_subject, t_label = data_target
        t_FC_feature = model.feature.feature(t_eeg).view(-1, data_size)
        t_FC_features[i * 32: (i + 1) * 32, :] = t_FC_feature.squeeze()  # 使用squeeze确保维度匹配

    print(s_FC_features.size())
    print(t_FC_features.size())

    torch.save(s_FC_features, os.path.join("record", f"{method}_s_data_stacked.pth"))
    torch.save(t_FC_features, os.path.join("record", f"{method}_t_data_stacked.pth"))

# 定义模型和交叉验证ID
cross_id = 0
# methods_list = ["Baseline", "DDC", "DeepCoral", "DANN", "DANNWass", "EEG_Infinity003", "EEG_Infinity004", "EEG_Infinity005Wass"]
methods_list = ["Baseline", "EEG_Infinity005Wass"]

# 对每个方法执行处理
for method in methods_list:
    process_method(method, cross_id, task="MengExp12ToMengExp3")
