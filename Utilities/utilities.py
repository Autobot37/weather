
def compute_number_of_points(extent, resolution):
    return int((extent[1] - extent[0])/resolution)


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
    

def save_rainydaysdata_file(data_dir, mask, method):
    if method == 0:
        grouped_files = rainy_days(data_dir, mask)
        with open('rainy_days.pkl', 'wb') as f:
            pickle.dump(grouped_files, f)
        print("Rainydays file saved")

    elif method == 1:
        grouped_files = rainy_days_plus_prev_days(data_dir, mask)
        with open('/home/vatsal/MOSDAC/rainy_days_plusprev_days.pkl', 'wb') as f:
            pickle.dump(grouped_files, f)
        print("Rainydays with previous days file saved")

    exit()