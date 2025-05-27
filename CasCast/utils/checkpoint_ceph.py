import imp
from torch.utils.data import Dataset

try:
    from petrel_client.client import Client
except:
    pass
from tqdm import tqdm
import numpy as np
import io
import torch
import os

class checkpoint_ceph(object):
    def __init__(self, conf_path="~/petreloss.conf", checkpoint_dir="weatherbench:s3://weatherbench/checkpoint") -> None:
        self.conf_path = conf_path
        # Convert S3-style path to local directory
        if "s3://" in checkpoint_dir:
            self.checkpoint_dir = "./checkpoints"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_checkpoint(self, url):
        file_path = os.path.join(self.checkpoint_dir, url)
        if not os.path.exists(file_path):
            return None
        checkpoint_data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        return checkpoint_data
    
    def load_checkpoint_with_ckptDir(self, url, ckpt_dir):
        file_path = os.path.join(ckpt_dir, url)
        if not os.path.exists(file_path):
            return None
        checkpoint_data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        return checkpoint_data
    
    def save_checkpoint(self, url, data):
        file_path = os.path.join(self.checkpoint_dir, url)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(data, file_path)

    def save_prediction_results(self, url, data):
        file_path = os.path.join(self.checkpoint_dir, url)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, data)