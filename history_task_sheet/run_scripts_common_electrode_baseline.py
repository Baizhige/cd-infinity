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

    # 构建并运行命令 (需要管理员权限)
    command = ["python", script_path] + cmd_args
    subprocess.run(command, check=True)

    # 恢复原工作目录
    os.chdir(current_dir)

# 脚本配置：每个元素是一个元组，包含脚本路径、工作目录和命令行参数
scripts_to_run = [
    ("common_electrode_baseline.py", ".", ["--config", "config_MengExp3ToBCICIV2A.ini", "--cache_prefix", "common_electrode_baseline_MengExp3ToBCICIV2A_1", "--prior_information", "1"]),
    ("common_electrode_baseline.py", ".", ["--config", "config_MengExp3ToMengExp12.ini", "--cache_prefix", "common_electrode_baseline_MengExp3ToMengExp12_1", "--prior_information", "1"]),
    ("common_electrode_baseline.py", ".", ["--config", "config_PhysioNetMIToMengExp3.ini", "--cache_prefix", "common_electrode_baseline_PhysioNetMIToMengExp3_1", "--prior_information", "1"]),
    ("common_electrode_baseline.py", ".", ["--config", "config_PhysioNetMIToMengExp12.ini", "--cache_prefix", "common_electrode_baseline_PhysioNetMIToMengExp12_1", "--prior_information", "1"]),
]
# 按顺序执行脚本
for script_path, work_dir, cmd_args in scripts_to_run:
    print(script_path, work_dir, cmd_args)
    run_script(script_path, work_dir, cmd_args)
