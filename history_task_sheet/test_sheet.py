import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA
import torch
print(torch.cuda.is_available())