from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import numpy as np
import configparser
import argparse
import os
from sklearn.metrics import accuracy_score
from my_utils.recorder import append_results_to_csv
from my_utils.EA_RA import EuclideanMeanCovariance
import torch


def z_score_normalization(data):
    """Z分数归一化"""
    return (data - np.mean(data, axis=2, keepdims=True)) / np.std(data, axis=2, keepdims=True)


parser = argparse.ArgumentParser(description='Read configuration file.')
parser.add_argument('--config', default='config_MengExp3ToBCICIV2A.ini', help='Path to the config.ini file')
parser.add_argument('--cache_prefix', default='parser_test2', help='prefix of the cache (IMPORTANT!)')
parser.add_argument('--prior_information', default='1', help='if the prior_information is used')

args = parser.parse_args()

config = configparser.ConfigParser()
config.read(os.path.join("hyperparameters", args.config))

cache_prefix = args.cache_prefix
NFold = config.getint('settings', 'NFold')
record_val_metric = np.zeros([2, NFold])
record_test_metric = np.zeros([2, NFold])
print(cache_prefix)

for cross_id in range(NFold):
    print("Cross validation {0}-fold".format(cross_id))
    # load data
    source_eeg_root = os.path.join(os.path.pardir, os.path.pardir, "EEGData")
    source_train_list = [os.path.join(config.get('settings', 'source_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'source_datafile_name')),
                         os.path.join(config.get('settings', 'source_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'source_labelfile_name'))]
    source_eval_list = [os.path.join(config.get('settings', 'source_path'), "concatedData", "eval",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'source_datafile_name')),
                        os.path.join(config.get('settings', 'source_path'), "concatedData", "eval",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'source_labelfile_name'))]
    source_test_list = [os.path.join(config.get('settings', 'source_path'), "concatedData", "test",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'source_datafile_name')),
                        os.path.join(config.get('settings', 'source_path'), "concatedData", "test",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'source_labelfile_name'))]

    target_eeg_root = os.path.join(os.path.pardir, os.path.pardir, "EEGData")
    target_train_list = [os.path.join(config.get('settings', 'target_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'target_datafile_name')),
                         os.path.join(config.get('settings', 'target_path'), "concatedData", "train",
                                      "cross_{0}_".format(cross_id) + config.get('settings', 'target_labelfile_name'))]
    target_eval_list = [os.path.join(config.get('settings', 'target_path'), "concatedData", "eval",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'target_datafile_name')),
                        os.path.join(config.get('settings', 'target_path'), "concatedData", "eval",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'target_labelfile_name'))]
    target_test_list = [os.path.join(config.get('settings', 'target_path'), "concatedData", "test",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'target_datafile_name')),
                        os.path.join(config.get('settings', 'target_path'), "concatedData", "test",
                                     "cross_{0}_".format(cross_id) + config.get('settings', 'target_labelfile_name'))]

    print("Source dataset")
    print(source_train_list)
    print(source_eval_list)
    print(source_test_list)
    print("Target dataset")
    print(target_train_list)
    print(target_eval_list)
    print(target_test_list)

    # transfer_matrix_source = np.eye(config.getint('settings', 'source_num_channel'))

    transfer_matrix_target = np.load(os.path.join('config', config.get('settings', 'file_name_transfer_matrix')))

    # 获取训练数据
    source_data_train = z_score_normalization(np.load(os.path.join(source_eeg_root, source_train_list[0])))
    source_data_eval = z_score_normalization(np.load(os.path.join(source_eeg_root, source_eval_list[0])))
    source_data_test = z_score_normalization(np.load(os.path.join(source_eeg_root, source_test_list[0])))

    source_label_train = np.load(os.path.join(source_eeg_root, source_train_list[1]))
    source_label_eval = np.load(os.path.join(source_eeg_root, source_eval_list[1]))
    source_label_test = np.load(os.path.join(source_eeg_root, source_test_list[1]))

    target_data_train = z_score_normalization(np.load(os.path.join(target_eeg_root, target_train_list[0]))[:300,:,:])
    target_data_eval = z_score_normalization(np.load(os.path.join(target_eeg_root, target_eval_list[0])))
    target_data_test = z_score_normalization(np.load(os.path.join(target_eeg_root, target_test_list[0])))

    # target_data_train = torch.tensor(transfer_matrix_target).cuda() @ torch.tensor(target_data_train).cuda()
    # target_data_eval = torch.tensor(transfer_matrix_target).cuda() @ torch.tensor(target_data_eval).cuda()
    # target_data_test = torch.tensor(transfer_matrix_target).cuda() @ torch.tensor(target_data_test).cuda()

    # target_data_train = target_data_train.detach().cpu().numpy()
    # target_data_eval = target_data_eval.detach().cpu().numpy()
    # target_data_test = target_data_test.detach().cpu().numpy()


    target_label_train = np.load(os.path.join(target_eeg_root, target_train_list[1]))[:300]
    target_label_eval = np.load(os.path.join(target_eeg_root, target_eval_list[1]))
    target_label_test = np.load(os.path.join(target_eeg_root, target_test_list[1]))

    # EA
    print("Operating EA")
    target_EA = EuclideanMeanCovariance(target_data_train, is_cuda=True)
    target_data_train = target_EA.transform(target_data_train).detach().cpu().numpy()
    target_data_eval = target_EA.transform(target_data_eval).detach().cpu().numpy()
    target_data_test = target_EA.transform(target_data_test).detach().cpu().numpy()
    print("EA Done")

    # 创建线性分类器
    lda = LinearDiscriminantAnalysis()
    # 创建CSP提取特征，这里使用 个分量的 CSP
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    # 创建机器学习的Pipeline,也就是分类模型，使用这种方式可以把特征提取和分类统一整合到了clf中
    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    print(target_data_train.shape)
    print(target_label_train.shape)
    clf.fit(target_data_train, target_label_train)
    print("==========")
    # 进行测试
    target_pred_train = clf.predict(target_data_train)
    target_accuracy_train = accuracy_score(target_label_train, target_pred_train)
    target_pred_eval = clf.predict(target_data_eval)
    target_accuracy_eval = accuracy_score(target_label_eval, target_pred_eval)

    target_pred_test = clf.predict(target_data_test)
    target_accuracy_test = accuracy_score(target_label_test, target_pred_test)

    print('Accuracy of the target train set: {0}'.format(target_accuracy_train))
    print('Accuracy of the target validation set: {0}'.format(target_accuracy_eval))
    record_val_metric[0, cross_id] = target_accuracy_eval
    print('Accuracy of the target test set: {0}'.format(target_accuracy_test))
    record_test_metric[0, cross_id] = target_accuracy_test
    # # EA
    # print("Operating EA")
    # source_EA = EuclideanMeanCovariance(source_data_train, is_cuda=True)
    # source_data_train = source_EA.transform(source_data_train).detach().cpu().numpy()
    # source_data_eval = source_EA.transform(source_data_eval).detach().cpu().numpy()
    # source_data_test = source_EA.transform(source_data_test).detach().cpu().numpy()
    # print("EA Done")
    #
    # # 创建线性分类器
    # lda = LinearDiscriminantAnalysis()
    # # 创建CSP提取特征，这里使用 个分量的 CSP
    # csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    # # 创建机器学习的Pipeline,也就是分类模型，使用这种方式可以把特征提取和分类统一整合到了clf中
    # clf = Pipeline([('CSP', csp), ('LDA', lda)])
    #
    #
    #
    # print(source_data_train.shape)
    # print(source_label_train.shape)
    # clf.fit(source_data_train, source_label_train)
    # print("==========")
    # # 进行测试
    # source_pred_train = clf.predict(source_data_train)
    # source_accuracy_train = accuracy_score(source_label_train, source_pred_train)
    # source_pred_eval = clf.predict(source_data_eval)
    # source_accuracy_eval = accuracy_score(source_label_eval, source_pred_eval)
    #
    # source_pred_test = clf.predict(source_data_test)
    # source_accuracy_test = accuracy_score(source_label_test, source_pred_test)
    #
    # print('Accuracy of the Source train set: {0}'.format(source_accuracy_train))
    # print('Accuracy of the Source validation set: {0}'.format(source_accuracy_eval))
    # record_val_metric[0, cross_id] = source_accuracy_eval
    # print('Accuracy of the Source test set: {0}'.format(source_accuracy_test))
    # record_test_metric[0, cross_id] = source_accuracy_test



np.save(os.path.join("record", cache_prefix + "_val_metric.npy"), record_val_metric)
np.save(os.path.join("record", cache_prefix + "_test_metric.npy"), record_test_metric)
print("============Final Summary==================================")
print("record_val_metric")
print(record_val_metric)
print(np.mean(record_val_metric, axis=1))
print("record_test_metric")
print(record_test_metric)
print(np.mean(record_test_metric, axis=1))
# append_results_to_csv(cache_prefix, record_val_metric, record_test_metric)
