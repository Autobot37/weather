from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim

from Utilities import *
import argparse
import wandb
from torchinfo import summary
from collections import defaultdict
# from dataset2 import *
import xarray as xr
from sklearn.model_selection import train_test_split
import pickle

# class RTimeSeriesDataset(Dataset):
#     def __init__(self, file_pairs,data_dir,mask, mean, std):
#         self.data_dir = data_dir
#         self.file_pairs = file_pairs  # List of (input_files, target_files)
#         self.mask = mask
#         self.std = std
#         self.mean = mean
#         # Precompute all file paths
#         self.all_files = set()
#         for inp_files, tar_files in file_pairs:
#             self.all_files.update(inp_files + tar_files)
        
#         # Preload metadata or cache frequently used files
#         self.cache = {} 

#     def __len__(self):
#         # Total samples minus the sequence lengths needed for input and output
#         # return self.num_sequences
#         return len(self.file_pairs)
        
#     def __getitem__(self, idx):
#         input_files, target_files = self.file_pairs[idx]
        
#         # Load input sequence
#         input_data = [self._load_file(f) for f in input_files]
        
#         # Load target sequence
#         target_data = [self._load_file(f) for f in target_files]
        
#         return torch.stack(input_data), torch.stack(target_data)
    
#     def _load_file(self, filename):
#         """Cached file loading with lazy preprocessing"""
#         if filename not in self.cache:
#             path = os.path.join(self.data_dir, filename)
#             data = xr.open_dataset(path)
#             data_array = data.data_vars['DBZ'][0, :, :] 
#             data_array = preprocessing_lstm(self.mask, data_array, self.mean, self.std)
#             self.cache[filename] = torch.tensor(data_array, dtype = torch.float32)
#         return self.cache[filename]
    
    # def __getitem__(self, idx):
    #     # Load input sequence (16 timesteps)

    #     input_data_files = self.files[idx][0]
    #     output_data_files = self.files[idx][1]

    #     input_data = []
    #     for file in input_data_files:
    #         file_path = os.path.join(self.data_dir, file)
    #         data = xr.open_dataset(file_path)
    #         input_data.append(torch.tensor(preprocessing(data, self.mask)))
def rainy_dataset(data_dir, input_seq_length, output_seq_length, mean = None, std = None):


    ds = xr.open_dataset("/home/vatsal/MOSDAC/train_test/full_dataset/05AUG2020_161736.nc", engine="netcdf4")
    mask = ds['DBZ'][0,:,:]>0

    grouped_files = {}

    train_dic = {}
    val_dic = {}
    test_dic = {}
    j=0

    # Create dictionary for rainy dataset  

    # grouped_files = rainy_days(data_dir, mask)
    # # Storing dictionary in disc

    # with open('rainy_days.pkl', 'wb') as f:
    #     pickle.dump(grouped_files, f)
    # exit()
    #Load the dictionay, no need to create again and again

    with open('/home/vatsal/MOSDAC/rainy_days.pkl', 'rb') as f:
        grouped_files = pickle.load(f)
    

    grouped_files = dict(list(grouped_files.items()))

    # For global mean and std an min and max

    # dataset_rain = None
    # for date, files in grouped_files.items():
    #     data_add = os.path.join(data_dir, files[0])
    #     data = xr.open_dataset(data_add, engine="netcdf4")
    #     data = data.data_vars['DBZ'][0, :, :].values
    #     data = preprocessing_lstm(mask, data)
    #     data = np.expand_dims(data, axis=0)  # shape: (1, H, W)
    
    #     if dataset_rain is None:
    #         dataset_rain = data
    #     else:
    #         dataset_rain = np.concatenate((dataset_rain, data), axis=0)

    #     print("Data shape", dataset_rain.shape)
    # min = np.min(dataset_rain, axis=0)
    # print("Min shape", mean.shape)
    # max = np.max(dataset_rain, axis=0)
    # print("Max shape", std.shape)
    # np.savez('data_stats2.npz', min=min, max=max)
    # exit()

    stop_key = "29SEP2023"

    train_files = {}

    for date, files in grouped_files.items():
    
        train_files[date] = files
        if date == stop_key:
            break
    # Validation and test files
    keys = list(grouped_files.keys())
    stop_index = keys.index(stop_key)

    remaining_items = list(grouped_files.items())[stop_index + 1:]
    # Step 2: Split into two halves
    mid = len(remaining_items) // 2

    validation_files = dict(remaining_items[:mid])
    test_files = dict(remaining_items[mid:])

    def generate_sequences(data_dict, input_seq_length, output_seq_length):
        seq_dict = {}
        index = 0  # Index for dictionary keys
        for date, files in data_dict.items():
            # print(date)
            if len(files) < input_seq_length + output_seq_length:
                continue
            for i in range(len(files) - (input_seq_length + output_seq_length - 1)):
                seq_dict[index] = (
                    files[i:i + input_seq_length], 
                    files[i + input_seq_length:i + input_seq_length + output_seq_length]
                )
                index += 1
        return seq_dict

    # print("Train")
    train_dic = generate_sequences(train_files, input_seq_length, output_seq_length)

    # print("Validation")
    val_dic = generate_sequences(validation_files, input_seq_length, output_seq_length)

    # print("Test")
    test_dic = generate_sequences(test_files, input_seq_length, output_seq_length)
    # Extract sequences
    train_files = list(train_dic.values())
    val_files = list(val_dic.values())
    test_files = list(test_dic.values())

    print(len(train_files))
    print(len(val_files))
    print(len(test_files))

    tf = []
    j=2
    for i in range(15):
        tf.append(test_files[j])
        j = j + 30
    
    train_dataset = RTimeSeriesDataset(train_files, data_dir,mask, mean, std)
    test_dataset = RTimeSeriesDataset(tf, data_dir, mask, mean, std)
    val_dataset = RTimeSeriesDataset(val_files, data_dir,mask,  mean, std)

    return train_dataset, test_dataset, val_dataset

def create_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,        # Parallel loading
        pin_memory=True,      # Faster GPU transfers
        persistent_workers=True,  # Maintain worker pool
        drop_last=True
    )
