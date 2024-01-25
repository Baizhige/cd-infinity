import subprocess
import os

def run_script(script_path, work_dir, cmd_args=[]):
    """
    在指定的工作目录下执行一个Python脚本，并传入命令行参数。

    :param script_path: Python脚本的路径
    :param work_dir: 工作目录的路径
    :param cmd_args: 要传递给脚本的命令行参数列表
    """
    # 保存当前工作目录
    current_dir = os.getcwd()

    # 改变工作目录
    os.chdir(work_dir)

    # 构建并运行命令
    command = ["python", script_path] + cmd_args
    subprocess.run(command, check=True)

    # 恢复原工作目录
    os.chdir(current_dir)

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

scripts_to_run = []
# 遍历所有任务和backbone类型
for task in tasks_list:
    for backbone_type in backbones_list:
        # 为每个任务和backbone类型组合创建一个元组，并添加到列表中
        scripts_to_run.append(
            ("EEG_Infinity003_anybackbone.py", ".",
             ["--config", f"config_{task}.ini", "--cache_prefix", f"Comparison_EEG_Infinity003_{backbone_type}_{task}_1",
              "--backbone_type", backbone_type, "--prior_information", "1"])
        )

# 按顺序执行脚本
for script_path, work_dir, cmd_args in scripts_to_run:
    print(script_path, work_dir, cmd_args)
    run_script(script_path, work_dir, cmd_args)
