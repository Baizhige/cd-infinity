# Task Deployment Framework

## Description

This repository provides a framework for deploying transfer learning tasks using a script-based engine. This project 
file has been reorganized for release purposes. Users will need to update certain environment paths before running it,
such as working environment, cuda setting, python executor path. The project structure is organized as follows:

- **`run_scripts_engine.py`**: Main entry point to run tasks, such as basedline, DANet, DANN, DANNWass, DDC, DeepCoral,
- and EEG-Infinity.
- **`record_archive/`**: Stores the original experiment records.
- **`task_scripts/`**: Contains individual scripts for different transfer learning methods.
- **`ini_config/`**: Holds configuration files for training tasks.
- **`dataset_config_files/`**: Includes temporary matrices and metadata used during transfer learning.

### Dataset Structure

The dataset should be located at `../../dataset/` and follow the expected folder hierarchy:

Each file within the dataset follows the naming convention: cross_<cross_id>_data_<sampling_rate>_<channel_number>_<pre_processing_pipeline>.npy

Due to GitHub limitations, if you need access to the dataset folder, please email: C.Qin8@liverpool.ac.uk.