import os
import glob
import re
from datetime import datetime
import pyart
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def convert_radartoxarray(f):
    z_lim, y_lim, x_lim = (0.,20000.),(-240500.,240500.),(-240500.,240500.)
    res_h, res_v = 1000, 245
    def npts(lim, res): return int((lim[1]-lim[0])/res)
    shape = (npts(z_lim, res_v), npts(y_lim, res_h), npts(x_lim, res_h))
    radar = pyart.io.read(f)
    radar.time['units'] = f"seconds since {extract_start_time(f)}"
    grid = pyart.map.grid_from_radars(
        radar, grid_shape=shape, grid_limits=(z_lim, y_lim, x_lim)
    )
    return grid.to_xarray()

def build_mask(ds, time_idx=0, height_idx=8):
    return ds['DBZ'][time_idx, height_idx].values > 0

def preprocess_dbz(arr, mask, min_val=-30.0, max_val=70.0):
    arr = np.where(mask, np.nan, arr) 
    arr[np.isnan(arr)] = min_val 
    arr = np.clip(arr, min_val, max_val)
    arr = (arr - min_val) / (max_val - min_val) #normalize to [0, 1]
    #also norming to [-1, 1]
    arr = arr * 2 - 1
    return arr

# def get_mean_std(all_files, mask):
#     if os.path.exists("mean_std.npz"):
#         data = np.load("mean_std.npz")
#         return data["mean"], data["std"]
#     cube = []
#     for f in tqdm(all_files, desc="Loading data for stats"):
#         arr = xr.open_dataset(f)["DBZ"].isel(time=0).values.astype(np.float32)
#         arr = np.where(mask, np.nan, arr)
#         arr[np.isnan(arr)] = 0
#         cube.append(arr)
#     cube = np.stack(cube, 0)
#     mean, std = cube.mean(0), cube.std(0)
#     np.savez("mean_std.npz", mean = mean, std = std)
#     return mean, std

class RadarNowcastDataset(Dataset):
    def __init__(self, files, cond_window, pred_window, mask):
        self.files = sorted(files, key=_parse_ts)
        self.cond_window = cond_window
        self.pred_window = pred_window
        self.window = cond_window + pred_window
        self.mask = mask
        assert len(self.files) > self.window
        ds = xr.open_dataset(self.files[0])
        self.lat = ds['lat']
        self.lon = ds['lon']
        self.lat = torch.from_numpy(self.lat.values)
        self.lon = torch.from_numpy(self.lon.values)

    def __len__(self):
        return len(self.files) - self.window

    def __getitem__(self, i):
        seq = self.files[i : i + self.window + 1]
        frames = []
        for f in seq:
            arr = xr.open_dataset(f)["DBZ"].isel(time=0).values.astype(np.float32)
            frames.append(preprocess_dbz(arr, self.mask))
        stk = np.stack(frames, 0)

        x = torch.from_numpy(stk[:self.cond_window])
        y = torch.from_numpy(stk[self.cond_window:self.window])

        _, H, W = x.shape
        factor = 2**6
        ph = (factor - H % factor) % factor
        pw = (factor - W % factor) % factor
        pad = (pw//2, pw - pw//2, ph//2, ph - ph//2)
        
        return {
            'input': F.pad(x, pad),
            'target': F.pad(y, pad),
            'lat': self.lat,
            'lon': self.lon,
        }

def plot_sample(sample, vmin=-30, vmax=70, label1="Last Input Frame", label2="Target Frame", save = False, name = None):
    x = sample['input'][-1].numpy()
    y = sample['target'][0].numpy()
    lat = sample['lat'].numpy()
    lon = sample['lon'].numpy()
    H, W = lat.shape
    pad_h = x.shape[0] - H
    pad_w = x.shape[1] - W
    
    top = pad_h // 2
    left = pad_w // 2
    x_cropped = x[top:top+H, left:left+W]
    y_cropped = y[top:top+H, left:left+W]

    x_cropped = (x_cropped + 1.0) / 2.0
    y_cropped = (y_cropped + 1.0) / 2.0
 
    x_cropped[x_cropped <= 0] = np.nan
    y_cropped[y_cropped <= 0] = np.nan

    x_cropped = x_cropped * (70)
    y_cropped = y_cropped * (70)
    plt.figure(figsize=(16, 6))
    
    plt.subplot(121)
    plt.pcolormesh(lon, lat, x_cropped, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Reflectivity (dBZ)")
    plt.title(label1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.subplot(122)
    plt.pcolormesh(lon, lat, y_cropped, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Reflectivity (dBZ)")
    plt.title(label2)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/sample_{name}.png", dpi=300)
    plt.show()


def plot_dbz_raw(sample, vmin=0, vmax=70):
    x = sample['input'][-1].numpy()
    y = sample['target'][0].numpy()
    
    H, W = 481, 481
    pad_h = x.shape[0] - H
    pad_w = x.shape[1] - W
    top = pad_h // 2
    left = pad_w // 2
    
    x_cropped = x[top:top+H, left:left+W]
    y_cropped = y[top:top+H, left:left+W]
    
    x_cropped = (x_cropped + 1.0) / 2.0
    y_cropped = (y_cropped + 1.0) / 2.0

    x_cropped[x_cropped <= 0] = np.nan
    y_cropped[y_cropped <= 0] = np.nan

    x_cropped = x_cropped * (70)
    y_cropped = y_cropped * (70)
    
    plt.figure(figsize=(16, 6))

    plt.subplot(121)
    plt.imshow(x_cropped, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Reflectivity (dBZ)")
    plt.title("Last Input Frame")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")

    plt.subplot(122)
    plt.imshow(y_cropped, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Reflectivity (dBZ)")
    plt.title("Target Frame")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")

    plt.tight_layout()
    plt.show()


    