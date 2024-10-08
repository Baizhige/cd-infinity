import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from .data_loader_npy import EEGDataSet
from .my_tool import confusion_matrix

def test(test_list, model_path, domain, start=0,num_channel=62,transforms=None):

    model_root = 'models'
    data_root = os.path.join(os.path.pardir,os.path.pardir,"EEGData")

    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    alpha = 0
    """load data"""

    dataset = EEGDataSet(
        data_root=data_root,
        data_list=test_list,
        start=start,
        num_channel=num_channel,
        datalen = 384,
    )


    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, model_path
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    conf_matrix = torch.zeros(8,8)
    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_eeg, t_subject, t_label = data_target
        batch_size = len(t_label)

        if cuda:
            t_eeg = t_eeg.cuda()
            t_label = t_label.cuda()

        class_output, _, _, _ = my_net(input_data=t_eeg, domain=domain, alpha=alpha)

        pred = class_output.data.max(1, keepdim=True)[1]

        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        conf_matrix = confusion_matrix(pred, labels=t_label,conf_matrix=conf_matrix)

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu,conf_matrix
