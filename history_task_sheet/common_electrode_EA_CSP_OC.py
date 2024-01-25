from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from mne.decoding import CSP
import numpy as np
import os
from sklearn.metrics import accuracy_score
from my_utils.EA_RA import EuclideanMeanCovariance


def z_score_normalization(data):
    """Z分数归一化"""
    return (data - np.mean(data, axis=2, keepdims=True)) / np.std(data, axis=2, keepdims=True)


# def z_score_normalization(data):
#     """Z分数归一化"""
#     return data

# 获取训练数据
target_train_list = ['data_aa.npy', 'label_aa.npy']
target_eval_list = ['data_av.npy', 'label_av.npy']
target_test_list = ['data_al.npy', 'label_al.npy']


target_eeg_root = os.path.join(os.path.pardir, os.path.pardir, "EEGData","dataset_BCIIIIIV2A","processedData")
target_data_train = z_score_normalization(np.load(os.path.join(target_eeg_root, target_train_list[0])).transpose((0,2,1))).astype(np.float64)
target_data_eval = z_score_normalization(np.load(os.path.join(target_eeg_root, target_eval_list[0])).transpose((0,2,1))).astype(np.float64)
target_data_test = z_score_normalization(np.load(os.path.join(target_eeg_root, target_test_list[0])).transpose((0,2,1))).astype(np.float64)

target_label_train = np.load(os.path.join(target_eeg_root, target_train_list[1])).flatten().astype(np.float64)
target_label_eval = np.load(os.path.join(target_eeg_root, target_eval_list[1])).flatten().astype(np.float64)
target_label_test = np.load(os.path.join(target_eeg_root, target_test_list[1])).flatten().astype(np.float64)

# EA
# print("Operating EA")
# target_EA = EuclideanMeanCovariance(target_data_train, is_cuda=True)
# target_EA_eval = EuclideanMeanCovariance(target_data_eval, is_cuda=True)
# target_EA_test = EuclideanMeanCovariance(target_data_test, is_cuda=True)
#
# target_data_train = target_EA.transform(target_data_train).detach().cpu().numpy()
# target_data_eval = target_EA_eval.transform(target_data_eval).detach().cpu().numpy()
# target_data_test = target_EA_test.transform(target_data_test).detach().cpu().numpy()
# print("EA Done")

# 创建线性分类器
lda = LinearDiscriminantAnalysis()
# svm = LinearSVC(dual="auto", random_state=0, tol=1e-5)
# 创建CSP提取特征，这里使用 个分量的 CSP
csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
# 创建机器学习的Pipeline,也就是分类模型，使用这种方式可以把特征提取和分类统一整合到了clf中
clf = Pipeline([('CSP', csp), ('Classifier', lda)])

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
print('Accuracy of the target test set: {0}'.format(target_accuracy_test))
