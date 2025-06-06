from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
torch.cuda.set_device("cuda:1")
from Utilities import *
import argparse
import wandb
from torchinfo import summary
from collections import defaultdict
# from dataset2 import *
import xarray as xr
from sklearn.model_selection import train_test_split
import pickle

class RTimeSeriesDataset(Dataset):
    def __init__(self, file_pairs,data_dir,mask):
        self.data_dir = data_dir
        self.file_pairs = file_pairs  # List of (input_files, target_files)
        self.mask = mask
        # Precompute all file paths
        self.all_files = set()
        for inp_files, tar_files in file_pairs:
            self.all_files.update(inp_files + tar_files)
        
        # Preload metadata or cache frequently used files
        self.cache = {} 

    def __len__(self):
        # Total samples minus the sequence lengths needed for input and output
        # return self.num_sequences
        return len(self.file_pairs)
        
    def __getitem__(self, idx):
        input_files, target_files = self.file_pairs[idx]
        
        # Load input sequence
        input_data = [self._load_file(f) for f in input_files]
        
        # Load target sequence
        target_data = [self._load_file(f) for f in target_files]
        
        return torch.stack(input_data), torch.stack(target_data)
    
    def _load_file(self, filename):
        """Cached file loading with lazy preprocessing"""
        if filename not in self.cache:
            path = os.path.join(self.data_dir, filename)
            data = xr.open_dataset(path)
            data_array = data.data_vars['DBZ'][0, :, :] 
            data_array = preprocessing_lstm(self.mask, data_array)
            self.cache[filename] = torch.tensor(data_array, dtype = torch.float32)
        return self.cache[filename]
    
    # def __getitem__(self, idx):
    #     # Load input sequence (16 timesteps)

    #     input_data_files = self.files[idx][0]
    #     output_data_files = self.files[idx][1]

    #     input_data = []
    #     for file in input_data_files:
    #         file_path = os.path.join(self.data_dir, file)
    #         data = xr.open_dataset(file_path)
    #         input_data.append(torch.tensor(preprocessing(data, self.mask)))
def rainy_dataset(data_dir, input_seq_length, output_seq_length):
    # input_seq_length = 10
    # output_seq_length = 10

    ds = xr.open_dataset("train_test/full_dataset/05AUG2020_161736.nc", engine="netcdf4")
    mask = ds['DBZ'][0,:,:]>0

    grouped_files = {}

    dic = {}
    j=0

    #Create dictionary for rainy dataset  

    # grouped_files = rainy_days(data_dir, mask)
    # Storing dictionary in disc

    # with open('rainy_days.pkl', 'wb') as f:
    #     pickle.dump(grouped_files, f)

    # print("Dataset created and stored in rainy_days.pkl")
    # exit()
    #Load the dictionay, no need to create again and again
    with open('/home/vatsal/MOSDAC/rainy_days.pkl', 'rb') as f:
        grouped_files = pickle.load(f)
  
    grouped_files = dict(list(grouped_files.items()))
    for date, files in grouped_files.items():
        if len(files) < input_seq_length+ output_seq_length:
            continue
        else:
            for i in range(len(files)- (input_seq_length + output_seq_length-1)):
                dic[j] = (files[i:i+input_seq_length], files[i+input_seq_length:i+input_seq_length+output_seq_length])
                j+=1 

    num_sequences = j

    idx_train_seq = int(num_sequences * 0.8)
    idx_val_seq = int(num_sequences * 0.9)


    keys = list(dic.keys())

    training_keys = keys[:idx_train_seq]
    validation_keys =  keys[idx_train_seq:idx_val_seq]
    test_keys = keys[idx_val_seq:]
    
    train_files = [dic[train_keys] for train_keys in training_keys]
    val_files = [dic[val_keys] for val_keys in validation_keys]
    test_files = [dic[t_keys] for t_keys in test_keys]

    # tf = []
    # j=2
    # for i in range(40):
    #     tf.append(test_files[j])
    #     j = j + 10
    # tf = tf[-1:]
    train_dataset = RTimeSeriesDataset(train_files, data_dir,mask)
    test_dataset = RTimeSeriesDataset(test_files, data_dir, mask)
    val_dataset = RTimeSeriesDataset(val_files, data_dir,mask)

    return train_dataset, test_dataset, val_dataset

def create_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,        # Parallel loading
        pin_memory=True,      # Faster GPU transfers
        persistent_workers=True  # Maintain worker pool
    )

