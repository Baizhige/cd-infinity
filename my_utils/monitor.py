import os
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.animation import FFMpegWriter


def get_alignment_head_parameters(cache_suffix, epoch, i, targetorsource, len_dataloader=95, path_root=os.path.join("..", "collect_data")):
    target_path = os.path.join(path_root,
                               cache_suffix + f'{epoch * len_dataloader + i}_0_{targetorsource}_alignment_head_parameters.pth')
    model_parameters = torch.load(target_path)
    channel_transfer_matrix = model_parameters['channel_transfer_matrix'].squeeze().cpu()
    domain_filter_conv_weight = model_parameters['domain_filter.conv.weight'].squeeze().cpu()
    return channel_transfer_matrix, domain_filter_conv_weight


def get_scalars_from_log(cache_suffix, scalar_name='Target Validation Set Accuracy', path_root='logs'):
    # 初始化向量来存储scalar的值
    err_t_label_values = []

    # 构建正确的路径
    log_path = os.path.join(path_root, cache_suffix + '_Cross_0')

    # 设置EventAccumulator
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()  # 读取事件文件
    events = event_acc.Scalars(scalar_name)
    for event in events:
        err_t_label_values.append(event.value)
    return err_t_label_values




cache_suffix = 'parser_test2'  # 需要根据您的设置调整
num_epochs =  200  # 假设有10个epochs
len_dataloader = 90  # 假设每个epoch有93次迭代

# 从TensorBoard日志中读取数据
# # target_val_acc = get_scalars_from_log(cache_suffix, 'Target Validation Set Accuracy')
# target_train_loss = get_scalars_from_log(cache_suffix, 'err_t_label')

# 准备动画的绘图
fig, ax = plt.subplots(2, 2, figsize=(10, 10))  # 创建一个2x2的图表

def update(frame):
    print(f"process:{100*frame/(num_epochs*len_dataloader)}%")
    epoch, i = divmod(frame, len_dataloader)

    # 加载alignment head参数
    source_channel_transfer_matrix, source_domain_filter_conv_weight = get_alignment_head_parameters(cache_suffix,
                                                                                                     epoch, i, 'source')
    target_channel_transfer_matrix, target_domain_filter_conv_weight = get_alignment_head_parameters(cache_suffix,
                                                                                                     epoch, i, 'target')

    # 更新图表
    ax[0, 0].clear()
    ax[0, 1].clear()
    ax[1, 0].clear()
    ax[1, 1].clear()

    ax[0, 0].imshow(source_channel_transfer_matrix.numpy(), cmap='hot', interpolation='nearest')
    ax[0, 1].imshow(target_channel_transfer_matrix.numpy(), cmap='hot', interpolation='nearest')
    ax[1, 0].plot(source_domain_filter_conv_weight.numpy())
    ax[1, 1].plot(target_domain_filter_conv_weight.numpy())
    ax[1, 0].set_ylim([-0.25, 0.25])
    ax[1, 1].set_ylim([-0.25, 0.25])

    ax[0, 0].set_title('Source Channel Transfer Matrix')
    ax[0, 1].set_title('Target Channel Transfer Matrix')
    ax[1, 0].set_title('Source Domain Filter Conv Weight')
    ax[1, 1].set_title('Target Domain Filter Conv Weight')

# writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
#
# total_frames = num_epochs * len_dataloader
# ani = FuncAnimation(fig, update, frames=total_frames, repeat=False)
# plt.tight_layout()
# ani.save(os.path.join('..', 'animation','alignment_head_animation4.mp4'), writer=writer)

source_channel_transfer_matrix, source_domain_filter_conv_weight = get_alignment_head_parameters(cache_suffix,
                                                                                                     10, 1, 'target', len_dataloader=90)
source_channel_transfer_matrix = source_channel_transfer_matrix.cpu().detach().numpy()

aa= 0