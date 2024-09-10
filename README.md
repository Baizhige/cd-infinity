# Description
To deploy the task, simply run `run_scripts_xxxxx.py`.  
The file `EEG_XXXXXXX_anybackbone.py` contains the loss and training configurations for the specific task.  
The `hyperparameters` folder stores training configurations for different tasks.  
The `config` folder contains matrices for transfer learning.

# Usage
Curently, the dataset is designed to located in `../../dataset/` and follows a pre-defined folder structure:

- `xxx_dataset/concated_data/concatedData/train/` (training dataset)
- `xxx_dataset/concated_data/concatedData/eval/` (validation dataset)
- `xxx_dataset/concated_data/concatedData/test/` (testing dataset, used only once during the entire study)

Each file within the dataset follows the naming convention:
`cross_<cross_id>_data_<sampling_rate>_<channel_number>_<pre_processing_pipeline>.npy`

Due to GitHub limitations, if you need access to the dataset folder, please email: [C.Qin8@liverpool.ac.uk](mailto:C.Qin8@liverpool.ac.uk).
