import subprocess
import os

PYTHON_EXECUTABLE = "/gpfs/work/int/chengxuanqin/conda_envs/EEG/bin/python"

def run_script(script_path, work_dir, cmd_args=None):
    """
    Execute a Python script in a specified working directory with optional command line arguments.

    :param script_path: Path to the Python script
    :param work_dir: Directory to run the script in
    :param cmd_args: List of command line arguments to pass to the script
    """
    if cmd_args is None:
        cmd_args = []

    original_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        command = [PYTHON_EXECUTABLE, script_path] + cmd_args
        subprocess.run(command, check=True)
    finally:
        os.chdir(original_dir)

tasks = [
    'MengExp12ToMengExp3', 'MengExp12ToBCICIV2A', 'MengExp12ToPhysioNetMI',
    'PhysioNetMIToMengExp3', 'PhysioNetMIToMengExp12', 'PhysioNetMIToBCICIV2A',
    'BCICIV2AToMengExp3', 'BCICIV2AToMengExp12', 'BCICIV2AToPhysioNetMI',
    'MengExp3ToBCICIV2A', 'MengExp3ToMengExp12', 'MengExp3ToPhysioNetMI'
]

backbones = ['EEGNet', 'ShallowConvNet', 'DeepConvNet', 'InceptionEEG']

scripts = [
    'Baseline.py', 'DANN.py', 'DANNWass.py', 'DDC.py',
    'DeepCoral.py', 'PSAT', 'EEG-Infinity005Wass.py'
]

jobs_to_run = []

for script in scripts:
    for task in tasks:
        for backbone in backbones:
            method_name = os.path.splitext(script)[0].split('_')[-1]
            cache_prefix = f"Comparison_{method_name}_{backbone}_{task}_1"
            cmd_args = [
                "--dataset_config_files", f"config_{task}.ini",
                "--cache_prefix", cache_prefix,
                "--backbone_type", backbone,
                "--prior_information", "1"
            ]
            jobs_to_run.append((script, ".", cmd_args))

# Execute all prepared script jobs
for script_path, work_dir, cmd_args in jobs_to_run:
    print(f"Running: {script_path} in {work_dir} with args {cmd_args}")
    run_script(script_path, work_dir, cmd_args)
