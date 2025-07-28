import os
import json
import torch
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
    os.makedirs(configs["output_dir"], exist_ok=True)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx