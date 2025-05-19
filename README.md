# Task Deployment Framework

## Description

This repository provides a framework for deploying transfer learning tasks using a script-based engine. This project 
file has been reorganized for release purposes. Users will need to update certain environment paths before running it.
The project structure is organized as follows:

- **`run_scripts_engine.py`**: Main entry point to run tasks.
- **`record_archive/`**: Stores the original experiment records.
- **`task_scripts/`**: Contains individual scripts for different transfer learning methods.
- **`ini_config/`**: Holds configuration files for training tasks.
- **`dataset_config_files/`**: Includes temporary matrices and metadata used during transfer learning.

### Dataset Structure

The dataset should be located at `../../dataset/` and follow the expected folder hierarchy:

