import subprocess
import os

def run_script(script_path, work_dir, cmd_args=[]):
    """
    Execute a Python script in a specified working directory and pass in command line arguments.

    :param script_path: Path to the Python script
    :param work_dir: Path to the working directory
    :param cmd_args: List of command line arguments to pass to the script
    """
    # cd to specified work dir
    current_dir = os.getcwd()
    os.chdir(work_dir)
    # execute command
    command = ["python", script_path] + cmd_args
    subprocess.run(command, check=True)
    # cd to current dir
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
# test for each task
for task in tasks_list:
    # test for each backbone
    for backbone_type in backbones_list:
        # combine them
        scripts_to_run.append(
            ("EEG_DANN_anybackbone.py", ".",
             ["--config", f"config_{task}.ini", "--cache_prefix", f"Comparison_DANN_{backbone_type}_{task}_1",
              "--backbone_type", backbone_type, "--prior_information", "1"])
        )

# run commands
for script_path, work_dir, cmd_args in scripts_to_run:
    print(script_path, work_dir, cmd_args)
    run_script(script_path, work_dir, cmd_args)
