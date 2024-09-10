import os
import sys
from torch.optim import SGD
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from my_utils.data_loader_npy import EEGDataSet
from my_utils.model_EEG_Infinity002API_any_backboneDANNWass import EEG_Infinity
from my_utils.test_MengData_new import test
from my_utils.my_tool import CustomLRScheduler, generate_normalized_tensor_eye
from my_utils.recorder import append_results_to_csv
from torch.utils.tensorboard import SummaryWriter
import configparser
import argparse


parser = argparse.ArgumentParser(description='Read configuration file.')
parser.add_argument('--config', default='config_PhysioNetMIToMengExp12.ini', help='Path to the config.ini file')
parser.add_argument('--cache_prefix', default='parser_test2', help='prefix of the cache (IMPORTANT!)')
parser.add_argument('--prior_information', default='1', help='if the prior_information is used')
parser.add_argument('--backbone_type', default='InceptionEEG', help='choose the backbone type for feature extractor: EEGNet,ShallowConvNet,DeepConvNet,InceptionEEG,EEGSym')

args = parser.parse_args()

config = configparser.ConfigParser()
config.read(os.path.join("hyperparameters", args.config))

cache_prefix = args.cache_prefix
NFold = config.getint('settings', 'NFold')
record_val_metric = np.zeros([2, NFold])
record_test_metric = np.zeros([2, NFold])
n_epoch = config.getint('settings', 'n_epoch')
print(cache_prefix)
for cross_id in range(NFold):
    print("Cross validation {0}-fold".format(cross_id))
    model_root = 'models'
    cuda = True
    writer = SummaryWriter(os.path.join('logs', cache_prefix + '_Cross_{0}'.format(cross_id)))
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

    # load model
    CAR_matrix_source = torch.eye(config.getint('settings', 'source_num_channel'))

    transfer_matrix_source = CAR_matrix_source.cuda()

    with torch.no_grad():
        transfer_matrix_source_inv = torch.inverse(transfer_matrix_source)
    _right_idx_ = torch.tensor(np.load(os.path.join("config", config.get('settings', 'right_idx')))-1).cuda()
    _left_idx_ = torch.tensor(np.load(os.path.join("config", config.get('settings', 'left_idx')))-1).cuda()

    CAR_matrix_target = torch.eye(config.getint('settings', 'target_num_channel'))
    transfer_matrix_target = torch.matmul(
        torch.tensor(np.load(os.path.join('config', config.get('settings', 'file_name_transfer_matrix')))).to(
            torch.float32), CAR_matrix_target).cuda()
    if args.prior_information == '1':
        my_net = EEG_Infinity(transfer_matrix_source, transfer_matrix_target, num_channels=config.getint('settings', 'source_num_channel'), FIR_order=17, backbone_type=args.backbone_type, right_idx=_right_idx_, left_idx=_left_idx_)
        print("prior_information used!")
    else:
        print("no prior_information used!")
        transfer_matrix_source_random = generate_normalized_tensor_eye(config.getint('settings', 'source_num_channel'),
                                                                       config.getint('settings', 'source_num_channel'))
        transfer_matrix_target_random = generate_normalized_tensor_eye(config.getint('settings', 'source_num_channel'),
                                                                       config.getint('settings', 'target_num_channel'))
        my_net = EEG_Infinity(transfer_matrix_source_random, transfer_matrix_target_random, num_channels=config.getint('settings', 'source_num_channel'), FIR_order=17, backbone_type=args.backbone_type, right_idx=_right_idx_, left_idx=_left_idx_)
    # setup optimizer
    len_dataloader = min(len(target_train_dataloader), len(source_train_dataloader))
    total_steps = n_epoch * len_dataloader
    optimizer = SGD(my_net.parameters(), lr=config.getfloat('optimizer', 'lr'),
                    momentum=config.getfloat('optimizer', 'momentum'))
    scheduler = CustomLRScheduler(optimizer, mu=config.getfloat('optimizer', 'mu'),
                                  alpha=config.getfloat('optimizer', 'alpha'),
                                  beta=config.getfloat('optimizer', 'beta'), total_steps=total_steps)

    # two negative log losses
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    my_LogSoftmax = torch.nn.LogSoftmax(dim=1)

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()
        my_LogSoftmax = my_LogSoftmax.cuda()

    # training
    best_acc_source = 0.0
    best_index_source = 0

    best_acc_target = 0.0
    best_index_target = 0

    for epoch in range(n_epoch):
        # for each epoch, do
        data_source_iter = iter(source_train_dataloader)
        data_target_iter = iter(target_train_dataloader)
        my_net.train()
        for i in range(len_dataloader):
            my_net.zero_grad()

            # p stands for prorgession, ranging from 0 to 1.
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader

            # parameter of GRL
            alpha = 2. / (1. + np.exp(config.getint('GRL', 'decay') * p)) - 1

            # forward source data
            data_source = next(data_source_iter)

            s_eeg, s_subject, s_label = data_source

            s_domain_label = torch.ones(len(s_label)).long()
            if cuda:
                s_eeg = s_eeg.cuda()
                s_label = s_label.cuda()
                s_domain_label = s_domain_label.cuda()

            s_class_output, s_domain_output, _, _ = my_net(input_data=s_eeg, domain=0, alpha=alpha)
            # cls loss for source domain
            err_s_label = loss_class(my_LogSoftmax(s_class_output), s_label.long())
            err_s_domain = s_domain_output.mean()

            # forward data from target domain
            data_target = next(data_target_iter)
            t_eeg, t_subject, t_label = data_target

            t_domain_label = torch.zeros(len(t_label)).long()

            if cuda:
                t_eeg = t_eeg.cuda()
                t_label = t_label.cuda()
                t_domain_label = t_domain_label.cuda()

            t_class_output, t_domain_output, _, _ = my_net(input_data=t_eeg, domain=1, alpha=alpha)

            err_t_label = loss_class(my_LogSoftmax(t_class_output), t_label.long())
            err_t_domain = t_domain_output.mean()*(-1)

            # begin to compute DANN-wass loss
            err_DANN = err_s_label + err_s_domain + err_t_domain
            err_DANN.backward()

            # clip gradients
            my_net.clip_gradients_domain_classifier()

            # update weights
            optimizer.step()
            scheduler.step(epoch * len_dataloader + i)
            if config.getint('debug', 'isdebug'):
                sys.stdout.write(
                    '\r epoch: %d, [iter: %d / all %d], err_s_class: %f, err_t_class:%f, err_s_domain: %f, err_t_domain: %f, l1_distance: %f, cov_distance: %f, gradient_channel: %f' \
                    % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(), err_t_label.data.cpu().numpy(),
                       err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item(), l1_distance.data.cpu().item(), cov_distance.data.cpu().item(), gradient_channel.data.cpu().item()))
                sys.stdout.flush()

            with torch.no_grad():
                writer.add_scalar('err_s_label', err_s_label, epoch * len_dataloader + i)
                writer.add_scalar('err_t_label', err_t_label, epoch * len_dataloader + i)
                writer.add_scalar('err_s_domain', err_s_domain, epoch * len_dataloader + i)
                writer.add_scalar('err_t_domain', err_t_domain, epoch * len_dataloader + i)
                writer.add_scalar('l1_distance', err_t_domain, epoch * len_dataloader + i)
                writer.add_scalar('cov_distance', err_t_domain, epoch * len_dataloader + i)

        print('\n')
        acc_source = test(test_list=source_eval_list, torch_model=my_net, domain=0, num_channel=config.getint('settings', 'source_num_channel'))
        print('Cross: %d, Epoch: %d. Accuracy of the %s validation set: %f' % (cross_id, epoch, "Source", acc_source))

        acc_target = test(test_list=target_eval_list, torch_model=my_net, domain=1, num_channel=config.getint('settings', 'target_num_channel'))
        print('Cross: %d, Epoch: %d. Accuracy of the %s validation set: %f' % (cross_id, epoch, "Target", acc_target))
        writer.add_scalar('Source Validation Set Accuracy', acc_source, epoch)
        writer.add_scalar('Target Validation Set Accuracy', acc_target, epoch)

        if acc_source > best_acc_source:
            best_acc_source = acc_source
            best_index_source = epoch
            torch.save(my_net,
                       os.path.join(model_root, cache_prefix + '_cross_id_{0}_best_source_model.pth'.format(cross_id)))

        if acc_target > best_acc_target:
            best_acc_target = acc_target
            best_index_target = epoch
            torch.save(my_net,
                       os.path.join(model_root, cache_prefix + '_cross_id_{0}_best_target_model.pth'.format(cross_id)))

    print('============ Summary ============= \n')
    print('\n')
    test_acc_source = test(test_list=source_test_list,
                                               torch_model=cache_prefix + '_cross_id_{0}_best_source_model.pth'.format(cross_id), domain=0,
                                               num_channel=config.getint('settings', 'source_num_channel'))
    print('Accuracy of the %s test set: %f' % ("Source", test_acc_source))
    test_acc_target = test(test_list=target_test_list,
                                               torch_model=cache_prefix + '_cross_id_{0}_best_target_model.pth'.format(cross_id), domain=1,
                                               num_channel=config.getint('settings', 'target_num_channel'))
    print('Accuracy of the %s test set: %f' % ("Target", test_acc_target))

    print('Accuracy of the Exp12(Source) validation set: {0} at {1}'.format(best_acc_source, best_index_source))
    print('Accuracy of the Exp3(Target) validation set: {0} at {1}'.format(best_acc_target, best_index_target))
    record_val_metric[0, cross_id] = best_acc_source
    record_val_metric[1, cross_id] = best_acc_target

    record_test_metric[0, cross_id] = test_acc_source
    record_test_metric[1, cross_id] = test_acc_target

np.save(os.path.join("record", cache_prefix + "_val_metric.npy"), record_val_metric)
np.save(os.path.join("record", cache_prefix + "_test_metric.npy"), record_test_metric)
print("============Final Summary==================================")
print("record_val_metric")
print(record_val_metric)
print(np.mean(record_val_metric, axis=1))
print("record_test_metric")
print(record_test_metric)
print(np.mean(record_test_metric, axis=1))
append_results_to_csv(cache_prefix, record_val_metric, record_test_metric, file_path=os.path.join("record", "comparison_study_DANNWass.csv"))
