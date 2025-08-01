import os
import json
import torch
import time
import random
import shutil
import numpy as np
from typing import Tuple, List, Dict

def read_config(config_file_path: str):
    try:
        with open(config_file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except:
        return FileNotFoundError("The config file is corrupted/absent")

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def generate_dirs(configs:dict) -> None:
    output_dir = configs["output_dir"]

    if os.path.exists(output_dir):
        print(f"Warning: '{output_dir}' already exists. Deleting its contents...")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

#Decorator to print time for each function execution
def calculate_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        return result, duration
    return wrapper


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_pytorch_optimizations():
    """
    Set PyTorch performance optimizations for better training and inference speed.
    This includes:
    - High precision matrix multiplication
    - CUDNN benchmark mode for faster convolutions
    - Non-deterministic mode for better performance
    """
    import torch
    from torch.utils.tensorboard import SummaryWriter
    
    torch.set_float32_matmul_precision('high')
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



