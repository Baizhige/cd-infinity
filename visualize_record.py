import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List
import matplotlib.ticker as mticker
import matplotlib.font_manager as font_manager



def print_metric(df):
    # 检查第0列是否为Baseline
    if df.columns[0] != 'Baseline':
        print("Warning: The first column is not 'Baseline'.")
        return

    # 获取总的task数量
    total_tasks = len(df)

    # 遍历除了Baseline之外的每个method
    for method in df.columns[1:]:
        # 找到method在哪些task上超越了Baseline
        outperform_tasks = df[df[method] > df['Baseline']].index.tolist()
        # 找到method在哪些task上fail Baseline

        no_outperform_tasks = df[df[method] <= df['Baseline']].index.tolist()


        # 计算超越Baseline的task数量
        num_outperform = len(outperform_tasks)

        # 格式化并打印信息
        tasks_formatted = ', '.join(outperform_tasks)
        if num_outperform > 0:
            print(f"{method} ({num_outperform}/{total_tasks}) outperforms baseline on {tasks_formatted}.")
        print(no_outperform_tasks)

plt.rc('font', family='Times New Roman')
def visualize_record_anybackbone(methods_list: List[str], tasks_list: List[str], selected_backbones: List[str], selected_metric: str):
    # 用于存储提取的数据
    data = {}

    # 遍历所有方法和backbones，读取相应的CSV文件并提取所需数据
    for method in methods_list:
        for backbone in selected_backbones:
            file_name = os.path.join("record", f"comparison_study_{method}.csv")
            if os.path.exists(file_name):
                df = pd.read_csv(file_name)
                for task in tasks_list:
                    # 构建筛选条件
                    filter_condition = df['Cache Prefix'].str.contains(f'{method}_{backbone}_{task}')
                    # 提取指定的指标数据
                    filtered_data = df.loc[filter_condition, selected_metric].str.extract(r'([0-9.]+)').astype(float)
                    # 存储数据
                    if not filtered_data.empty:
                        data[(method, task, backbone)] = filtered_data.iloc[0, 0]
                    else:
                        data[(method, task, backbone)] = None

    # 将数据转换为DataFrame以便于绘图
    plot_data = pd.DataFrame(data.values(), index=pd.MultiIndex.from_tuples(data.keys()), columns=[selected_metric])
    plot_data = plot_data.unstack(level=-1)  # 将backbone提取为列

    # 删除metric层级，保留backbone层级
    plot_data = plot_data[selected_metric]

    # 设置绘图布局
    fig, axes = plt.subplots(len(selected_backbones), 1, figsize=(5, 3 * len(selected_backbones)), sharex=True)

    # 如果只有一个backbone，axes不是数组，这里进行处理
    if len(selected_backbones) == 1:
        axes = [axes]

    # 绘制每个backbone的子图
    for i, backbone in enumerate(selected_backbones):
        ax = axes[i]
        # 检查backbone是否存在于列中
        if backbone in plot_data.columns:
            # 筛选特定backbone的数据
            backbone_data = plot_data[backbone].unstack(level=0)
            # 绘制子图
            backbone_data = backbone_data[methods_list]
            print_metric(backbone_data)
            backbone_data.plot(kind='bar', ax=ax, legend=False, colormap='rainbow', linewidth=1)  # 只在第一个子图显示图例

        # 设置子图的标题为backbone的名称
        ax.set_title(backbone)

        # 设置y轴主刻度
        if i==0:
            ax.set_ylim(0.1, 0.85)
        else:
            ax.set_ylim(0.1, 0.85)
        # ax.set_ylabel(selected_metric, fontname='Times New Roman')
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # 设置y轴次刻度
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        # 设置网格线（仅主刻度）
        ax.grid(axis='y')

        # 设置X轴标签倾斜
        ax.tick_params(axis='x', rotation=90)

        # 设置字体为 Times New Roman
        #for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        #    label.set_fontname('Times New Roman')

    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)

    # 设置整体图例
    if len(selected_backbones) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',ncol=4)

    # 保存图片
    plt.savefig(os.path.join('figures', 'comparison_chart.png'), dpi=300, bbox_inches='tight')
# Example usage

tasks_list = ['BCICIV2AToMengExp3',
              'BCICIV2AToMengExp12',
              'BCICIV2AToPhysioNetMI',

              'MengExp3ToBCICIV2A',
              'MengExp3ToMengExp12',
              'MengExp3ToPhysioNetMI',

              'MengExp12ToBCICIV2A',
              'MengExp12ToMengExp3',
              'MengExp12ToPhysioNetMI',

              'PhysioNetMIToBCICIV2A',
              'PhysioNetMIToMengExp3',
              'PhysioNetMIToMengExp12',
              ]

backbones_list = ['EEGNet',
                  'ShallowConvNet',
                  'DeepConvNet',
                  'InceptionEEG']
# visualize_record(methods_list=["Baseline", "EA","RA","DDC","DeepCoral","EEG_Infinity003"], tasks_list=tasks_list, selected_backbone="EEGNet", selected_metric="Target Domain Test Accuracy")
# visualize_record_anybackbone(methods_list=["Baseline","DDC","DeepCoral","DANN","DANNWass", "EEG_Infinity003","EEG_Infinity005Wass"], tasks_list=tasks_list, selected_backbones=backbones_list, selected_metric="Target Domain Test Accuracy")
visualize_record_anybackbone(methods_list=["Baseline","DDC","DeepCoral","DANN","DANNWass","EEG_Infinity005Wass"], tasks_list=tasks_list, selected_backbones=backbones_list, selected_metric="Target Domain Test Accuracy")
# visualize_record_anybackbone(methods_list=["Baseline","EEG_Infinity003","EEG_Infinity004","EEG_Infinity005Wass", "EEG_Infinity006Wass"], tasks_list=tasks_list, selected_backbones=backbones_list, selected_metric="Target Domain Test Accuracy")
