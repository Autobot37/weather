import os
from glob import glob
import xarray as xr
import numpy as np
import pyart
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re
# Create a PyTorch Dataset class to return (input, target) pairs.

class ReflectivityDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data  # List of tensors
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)  # Number of sequences

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]  

def extract_date(filename):
    """Extracts the date part (DDMMMYYYY) from the filename."""
    match = re.match(r"(\d{2}[A-Z]{3}\d{4})_(\d{6})\.nc", filename)
    return match.group(1) if match else None

def sort_nc_files(folder_path):
    """Sorts files in chronological order based on their filenames."""
    files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]
    
    def extract_sort_key(filename):
        date_part, time_part = filename.split('_')
        months = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
                  'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
        day, month, year = date_part[:2], months[date_part[2:5]], date_part[5:]
        return f"{year}{month}{day}{time_part[:-3]}"  # YYYYMMDDHHMMSS

    return sorted(files, key=extract_sort_key)

def preprocessing_lstm(mask,ds):
    
    # reflec = ds['DBZ'][0,:,:]
    reflec = ds
    # Preprocessing 
    reflec = np.where(mask,np.nan, reflec )[:480, :480]
    np_reflec = np.array(reflec)
    np_reflec[np.isnan(np_reflec)] = 0
    return np.clip(np_reflec, 0, 50)

def rainy_days(folder_path, mask):

    grouped_files = defaultdict(list)
    spatial_means = defaultdict(list)
    # num_true_values = np.sum(mask.values)  # Count of True values

    # Iterate through sorted files in the folder
    for filename in sort_nc_files(folder_path):
        date_part = filename[:9]  # Extract date (DDMMMYYYY)
        file_path = os.path.join(folder_path, filename)

        # Open dataset and process the DBZ variable
        try:
            dataset = xr.open_dataset(file_path, engine="netcdf4")
            data_array = dataset.data_vars['DBZ'][0, :, :]  # Extract first time step
            data_array = preprocessing_lstm(mask, data_array)  # Apply preprocessing
            spatial_mean = np.mean(data_array)  # Compute spatial mean
            print(spatial_mean)
            # Store results
            grouped_files[date_part].append(filename)
            spatial_means[date_part].append(spatial_mean)

        except KeyError as e:
            print(f"KeyError: {e} in file {filename}. Skipping this file.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}. Skipping this file.")
        finally:
            # Ensure dataset is closed to free memory
            dataset.close()

    # Convert defaultdicts to regular dictionaries
    grouped_files = dict(grouped_files)
    spatial_means = dict(spatial_means)

    # Flatten all spatial means and compute global statistics
    flattened_sp_means = np.concatenate(list(spatial_means.values()))
    mean_of_means = np.mean(flattened_sp_means)
    std_of_means = np.std(flattened_sp_means)

    print(f"Overall Mean: {mean_of_means}")
    print(f"Overall Standard Deviation: {std_of_means}")

    # Identify rainy days based on threshold
    rainy_days_ = defaultdict(list)
    for date, mean_values in spatial_means.items():
        if np.mean(mean_values) > mean_of_means:  # Threshold: Mean of all spatial means
            rainy_days_[date] = grouped_files[date]

    print("Rainy Days:")
    for date, files in rainy_days_.items():
        print(f"{date}: {files}")

    return dict(rainy_days_)

def compute_number_of_points(extent, resolution):
    return int((extent[1] - extent[0])/resolution)

def minmaxscalingtensor(tensor, min, max):
    tensor_min =min
    tensor_max =max

    # Apply Min-Max Scaling
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return scaled_tensor

def glob_min_max(files):
    # Give all files
    ds = convert_radartoxarray("/home/vatsal/MOSDAC/RCTLS_05AUG2020_161736_L2B_STD.nc")
    mask = ds['DBZ'][0,8,:,:]>0

    all_data = []
    for file in files:
        ds = xr.open_dataset(file)
        reflec = ds['DBZ'][0,:,:]
        reflec = np.where(mask,np.nan, reflec )[:480, :480]
        np_reflec = np.array(reflec)
        np_reflec[np.isnan(np_reflec)] = 0
        all_data.append(np_reflec)
    stacked_all_data = np.stack(all_data, axis=0)
    min = stacked_all_data.min()
    max = stacked_all_data.max()
    with open("min_value.txt", "w") as file:
        file.write(str(min)) 
    with open("max_value.txt", "w") as file:
        file.write(str(min)) 


def extract_start_time(input_file):
    radar_x = xr.open_dataset(input_file, decode_times=False)
    array = radar_x['time_coverage_start'].values[0:17]
    decoded_string = b''.join(array).decode('utf-8')
    new_s = decoded_string.replace("Z", "") 

    
    formatted_datetime = datetime.strptime(new_s, "%Y-%m-%dT%H:%M:%S")

    # Convert back to string in the desired format
    final_string = formatted_datetime.strftime("%Y-%m-%dT%H:%M:%S")
    return final_string


def convert_radartoxarray(file_add):
    z_grid_limits = (0.,20000.)
    y_grid_limits = (-240500.,240500.)
    x_grid_limits = (-240500.,240500.)

    grid_resolutionh = 1000
    grid_resolutionv  = 245

    # Calculate the number of grid points
    x_grid_points = compute_number_of_points(x_grid_limits, grid_resolutionh)
    y_grid_points = compute_number_of_points(y_grid_limits, grid_resolutionh)
    z_grid_points = compute_number_of_points(z_grid_limits, grid_resolutionv)

    #Reading data
    radar_data = pyart.io.read(file_add)
    radar_data.time['units'] = f"seconds since {extract_start_time(file_add)}"

    #Height index
    

    #Converting the dataset to grid
    grid = pyart.map.grid_from_radars(radar_data, grid_shape = (z_grid_points, y_grid_points, x_grid_points), grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits))
    ds = grid.to_xarray()
    return ds


def cal_lon_range(dir):
    
    z_grid_limits = (0.,20000.)
    y_grid_limits = (-240500.,240500.)
    x_grid_limits = (-240500.,240500.)

    grid_resolutionh = 1000
    grid_resolutionv  = 245

    # Calculate the number of grid points
    x_grid_points = compute_number_of_points(x_grid_limits, grid_resolutionh)
    y_grid_points = compute_number_of_points(y_grid_limits, grid_resolutionh)
    z_grid_points = compute_number_of_points(z_grid_limits, grid_resolutionv)
    radar_data = pyart.io.read(dir)
    radar_data.time['units'] = f"seconds since {extract_start_time(dir)}"
    #Converting the dataset to grid
    grid = pyart.map.grid_from_radars(radar_data, grid_shape = (z_grid_points, y_grid_points, x_grid_points), grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits))
    ds = grid.to_xarray()
    lon_range = []
    for i in range(compute_number_of_points(x_grid_limits, grid_resolutionh)):
        reflec2 = ds.DBZ[0,8, 72,i]
        lon_range.append(float(reflec2['lon'].values))
    return lon_range


def cal_lat_range(dir):
    z_grid_limits = (0.,20000.)
    y_grid_limits = (-240500.,240500.)
    x_grid_limits = (-240500.,240500.)

    grid_resolutionh = 1000
    grid_resolutionv  = 245

    # Calculate the number of grid points
    x_grid_points = compute_number_of_points(x_grid_limits, grid_resolutionh)
    y_grid_points = compute_number_of_points(y_grid_limits, grid_resolutionh)
    z_grid_points = compute_number_of_points(z_grid_limits, grid_resolutionv)

    radar_data = pyart.io.read(dir)
    radar_data.time['units'] = f"seconds since {extract_start_time(dir)}"
    #Converting the dataset to grid
    grid = pyart.map.grid_from_radars(radar_data, grid_shape = (z_grid_points, y_grid_points, x_grid_points), grid_limits = (z_grid_limits, y_grid_limits, x_grid_limits))
    ds = grid.to_xarray()
    lat_range = []
    for i in range(compute_number_of_points(x_grid_limits, grid_resolutionh)):
        reflec2 = ds.DBZ[0,8, i,45]
        lat_range.append(float(reflec2['lat'].values))
    return lat_range


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if loss improves
        else:
            self.counter += 1  # Increase counter if no improvement

        return self.counter >= self.patience  # Stop training if patience exceeded
    

    