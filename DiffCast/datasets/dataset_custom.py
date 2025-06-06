import os
import glob
import re
import pickle
from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from termcolor import colored
import pyart

_ts_pattern = re.compile(r"(\d{2}[A-Z]{3}\d{4}_\d{6})")
def _parse_ts(fp: str) -> datetime:
    name = os.path.basename(fp).split(".")[0]
    m = _ts_pattern.search(name)
    if not m:
        raise ValueError(f"Filename `{name}` missing timestamp")
    return datetime.strptime(m.group(1), "%d%b%Y_%H%M%S")

def extract_start_time(f):
    arr = xr.open_dataset(f, decode_times=False)['time_coverage_start'].values[:17]
    s = b''.join(arr).decode("utf-8").rstrip("Z")
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

def build_mask(file, time_idx=0, height_idx=8):
    ds = xr.open_dataset(file, decode_times=False)
    mask = ds['DBZ'][time_idx, height_idx].values > 0
    ds.close()
    return mask

def preprocess_dbz(arr, mask, min_val, max_val):
    arr = np.where(mask, np.nan, arr) 
    arr[np.isnan(arr)] = min_val 
    arr = np.clip(arr, min_val, max_val)
    return arr

def calculate_mean_std(files, min_val, max_val, mask):
    global_mean = 0.0
    global_std = 0.0
    all_file_means = []

    for i, f in enumerate(tqdm(files, desc="Calculating mean and std")):
        ds = xr.open_dataset(f, engine='netcdf4')
        data_array = ds["DBZ"][0].values.astype(np.float32)
        ds.close()
        data_array = preprocess_dbz(data_array, mask, min_val, max_val)
        _mean = data_array.mean().item()
        _std = data_array.std().item()
        all_file_means.append(_mean)
        global_mean += _mean
        global_std += _std
    global_mean /= len(files)
    global_std /= len(files)
    with open("mean_std.pkl", "wb") as f:
        pickle.dump((global_mean, global_std, min_val, max_val), f)
    with open("all_file_means.pkl", "wb") as f:
        pickle.dump(all_file_means, f)
    return global_mean, global_std

class RadarNowcastDataset(Dataset):
    def __init__(self, path, seq_len, min_val = 0.0, max_val = 50.0, mask_file=None, img_size=480, split="train", filter_sparse=False, scaling = "01"):
        super().__init__()
        if not os.path.exists(path) or not os.path.isdir(path):
            print(colored(f"Path {path} does not exist or is not a directory.", "red"))
            raise FileNotFoundError(f"Path {path} does not exist or is not a directory.")

        files = sorted(glob.glob(os.path.join(path, "*.nc")))
        self.files = sorted(files, key=_parse_ts)
        self.window = seq_len
        self.mask = build_mask(mask_file)
        self.min_val = min_val
        self.max_val = max_val
        self.scaling = scaling

        # windowx idx  0, 1, 2, 
        with xr.open_dataset(self.files[0], engine='netcdf4') as ds:
            self.lat = torch.from_numpy(ds['lat'].values)
            self.lon = torch.from_numpy(ds['lon'].values)

        assert len(self.files) > self.window

        mean_std_pkl = "mean_std.pkl"
        if os.path.exists(mean_std_pkl):
            with open(mean_std_pkl, 'rb') as f:
                self.mean, self.std, pkl_min_val, pkl_max_val = pickle.load(f)
                assert pkl_min_val == min_val and pkl_max_val == max_val, "Min and Max values do not match with the precomputed values."
        else:
            self.mean, self.std = calculate_mean_std(self.files, self.min_val, self.max_val, self.mask)

        if filter_sparse:
            all_files_mean = pickle.load(open("all_file_means.pkl", "rb"))
            idxs = [idx for idx, val in enumerate(all_files_mean) if val >= self.mean]
            self.files = [self.files[i] for i in idxs]
            print(colored(f"Filtered {len(files) - len(self.files)} files based on mean reflectivity.", "green"))

        n = 0.8
        if split == "train":
            self.files = self.files[:int(len(self.files) * n)]
        else:
            self.files = self.files[int(len(self.files) * n):]

    def __len__(self):
        return min(2000, len(self.files) - self.window + 1)

    def __getitem__(self, i):
        seq = self.files[i : i + self.window]
        frames = []
        for f in seq:
            ds = xr.open_dataset(f, engine='netcdf4')
            arr = ds["DBZ"][0].values.astype(np.float32)
            ds.close()
            arr = preprocess_dbz(arr, self.mask, min_val = self.min_val, max_val = self.max_val)
            if self.scaling == "01":
                arr = (arr - self.min_val) / (self.max_val - self.min_val)
            elif self.scaling == "mean_std":
                arr = (arr - self.mean) / self.std
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling}")
            frames.append(arr)
        
        stk = np.stack(frames, 0)
        stk = torch.from_numpy(stk).unsqueeze(1)
        return stk
    
    def plot_sample(self, x, y = None, label1="Last Input Frame", label2="Target Frame", save = False, name = None):    
        #x, y = expected to be B, 1, H, W Numpy arrays
        plt.figure(figsize=(16, 6))
        plt.subplot(121)
        for i in range(x.shape[0]):
            plt.pcolormesh(self.lon, self.lat, x[i, 0], cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label="Reflectivity (dBZ)")
        plt.title(label1)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        if y is not None:
            plt.subplot(122)
            for i in range(y.shape[0]):
                plt.pcolormesh(self.lon, self.lat, y[i, 0], cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label="Reflectivity (dBZ)")
            plt.title(label2)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

        plt.tight_layout()
        if save:
            os.makedirs("custom_plots", exist_ok=True)
            plt.savefig(f"custom_plots/sample_{name}.png", dpi=300)
        plt.show()

    def plot_dbz_raw(self, x, y):        
        plt.figure(figsize=(16, 6))

        plt.subplot(121)
        plt.imshow(x, cmap='viridis')
        plt.colorbar(label="Reflectivity (dBZ)")
        plt.title("Last Input Frame")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")

        plt.subplot(122)
        plt.imshow(y, cmap='viridis')
        plt.colorbar(label="Reflectivity (dBZ)")
        plt.title("Target Frame")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    mask_file = "RCTLS_05AUG2020_161736_L2B_STD.nc"
    files_path = "/home/vatsal/NWM/custom/full_dataset"
    dataset = RadarNowcastDataset(files_path, seq_len=16, min_val=0.0, max_val=50.0, mask_file=mask_file, img_size=480, split="train", filter_sparse = False, scaling="01")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in tqdm(dataloader, desc="Loading data"):
        batch = batch.float()
        print(batch.shape)
        print(batch.min(), batch.max())
        break
    



"""

"""