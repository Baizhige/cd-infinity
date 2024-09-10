import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List
import matplotlib.ticker as mticker
import matplotlib.font_manager as font_manager



def print_metric(df):
    # Check if column 0 is the Baseline
    if df.columns[0] != 'Baseline':
        print("Warning: The first column is not 'Baseline'.")
        return

    total_tasks = len(df)

    # Traverse each method except Baseline
    for method in df.columns[1:]:
        # Find out on which tasks the method surpasses the baseline
        outperform_tasks = df[df[method] > df['Baseline']].index.tolist()
        # Find out on which tasks the method fails Baseline

        no_outperform_tasks = df[df[method] <= df['Baseline']].index.tolist()


        # Calculate the number of tasks that exceed the Baseline
        num_outperform = len(outperform_tasks)

        # print
        tasks_formatted = ', '.join(outperform_tasks)
        if num_outperform > 0:
            print(f"{method} ({num_outperform}/{total_tasks}) outperforms baseline on {tasks_formatted}.")
        print(no_outperform_tasks)

plt.rc('font', family='Times New Roman')
def visualize_record_anybackbone(methods_list: List[str], tasks_list: List[str], selected_backbones: List[str], selected_metric: str):
    # container
    data = {}

    # Traverse all methods and backbones, read the corresponding CSV file and extract the required data
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

    plot_data = pd.DataFrame(data.values(), index=pd.MultiIndex.from_tuples(data.keys()), columns=[selected_metric])
    plot_data = plot_data.unstack(level=-1)  # 将backbone提取为列

    # Delete the metric level and keep the backbone level
    plot_data = plot_data[selected_metric]

    fig, axes = plt.subplots(len(selected_backbones), 1, figsize=(5, 3 * len(selected_backbones)), sharex=True)

    # If there is only one backbone, axes is not an array, so process it here
    if len(selected_backbones) == 1:
        axes = [axes]

    for i, backbone in enumerate(selected_backbones):
        ax = axes[i]
        if backbone in plot_data.columns:
            # Filter data for a specific backbone
            backbone_data = plot_data[backbone].unstack(level=0)
            backbone_data = backbone_data[methods_list]
            print_metric(backbone_data)
            backbone_data.plot(kind='bar', ax=ax, legend=False, colormap='rainbow', linewidth=1)  # 只在第一个子图显示图例

        ax.set_title(backbone)

        if i==0:
            ax.set_ylim(0.1, 0.85)
        else:
            ax.set_ylim(0.1, 0.85)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


        ax.grid(axis='y')

        ax.tick_params(axis='x', rotation=90)



    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)

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
# visualize_record_anybackbone(methods_list=["Baseline","DDC","DeepCoral","DANN","DANNWass","EEG_Infinity005Wass"], tasks_list=tasks_list, selected_backbones=backbones_list, selected_metric="Target Domain Test Accuracy")
visualize_record_anybackbone(methods_list=["Baseline","EEG_Infinity003","EEG_Infinity004","EEG_Infinity005Wass", "EEG_Infinity006Wass"], tasks_list=tasks_list, selected_backbones=backbones_list, selected_metric="Target Domain Test Accuracy")
