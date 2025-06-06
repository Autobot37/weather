import os 
from conv_lstm import *
# from neuralop.models import FNO
import torch
import numpy as np
from Utilities import *
from Rainy_Dataset import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import pyart

model_weights = ['ConvLSTM_SGD_0.0001_3_2_32.pth']
dir_add = "/home/vatsal/MOSDAC/model_weights_nonormalization/"

hidden_dim = []
kernel_size_ = []
hidden_channels = 32
kernel_size = 3
for i in range(2):
    kernel = []
    hidden_dim.append(hidden_channels)
    kernel.append(kernel_size)
    kernel.append(kernel_size)
    kernel_size_.append(tuple(kernel))



# Define the ConvLSTM model
model = ConvLSTMPredictor(
    input_dim=1,       
    hidden_dims = hidden_dim,   # Hidden channels per layer (2 layers)
    kernel_size = kernel_size_,     # Convolution kernel size
    batch_first=True,       # Input shape: (batch, time, channels, H, W)
    bias=True,
    return_all_layers=False # Return only the last layer's output
)

lat_range = cal_lat_range("/home/vatsal/MOSDAC/RCTLS_05AUG2020_161736_L2B_STD.nc")
lon_range = cal_lon_range("/home/vatsal/MOSDAC/RCTLS_05AUG2020_161736_L2B_STD.nc")
lats_data = lat_range
lons_data = lon_range

data_dir = '/home/vatsal/MOSDAC/train_test/full_dataset/'
train_dataset, test_dataset, val_dataset = rainy_dataset(data_dir, 10, 10)

# Setting some incode parameters
start_idx = 0
num_slices = 5

for weight_add in model_weights:
    path = os.path.join(dir_add, weight_add)
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    x_test_lis = []
    y_test_lis = []
    y_pred_lis = []

    with torch.no_grad():  # Disable gradient computation for inference
        for i, (x_test, y_test) in enumerate(test_loader):
            print(i)
            x_test = x_test.unsqueeze(2)  # Move input to GPU

            pred = model(x_test).squeeze(2)  # Forward pass
            
           
            x_test = x_test.squeeze(2)
            # Move prediction back to CPU and convert to numpy

            x_test, pred = x_test.cpu().numpy(), pred.cpu().numpy() 
            pred = np.where((pred<0), 0, pred) 
            y_test = y_test.numpy()
            print(f"Predicted shape: {pred.shape}")
            print(f"Test input shape:{x_test.shape}")
            print(f"Test o/p shape:{y_test.shape}")

            x_test_lis.append(x_test)
            y_test_lis.append(y_test)
            y_pred_lis.append(pred)

    
    

            # Resizing for my dataset
            lon_grid, lat_grid = np.meshgrid(lons_data, lats_data)
            lon_grid = lon_grid[0:480, 0:480]
            lat_grid = lat_grid[0:480, 0:480]
        

            
            # Finding pred and gt file address
            # x_add = os.path.join(dir, x[i])
            # gt_add = os.path.join(dir, gt_file[i])
            # pred_add = os.path.join(dir, pred_file[i])


            # ground_truth = np.load(gt_add)
            # predicted = np.load(pred_add)
            # x_test = np.load(x_add)


            # 0 because there is batch size also
            x_test = x_test[0]
         
            ground_truth = y_test[0]
            predicted = pred[0]
            persistence = x_test[-1]
            
            difference = ground_truth - predicted
            # difference_p = []
            # for i in range(ground_truth.shape()[0]):
            #     diff = ground_truth[i] - persistence
            #     difference_p.append(diff)
            # difference_p = np.stack(difference_p, axis = 0)

            difference_p = ground_truth - persistence  # Shape remains (channels, x, y)
        
            end_idx = start_idx + num_slices

            start_idx2= end_idx
            end_idx2 = start_idx2+num_slices


            fig1, axes1 = plt.subplots(10, num_slices, figsize=(40, 40), subplot_kw={"projection": ccrs.PlateCarree()})
            # fig2, axes2 = plt.subplots(5,num_slices, figsize=(20, 20), subplot_kw={"projection": ccrs.PlateCarree()})
            colors = [(0, "blue"), (0.5, "lightgray"), (1, "red")]
            cmap_diverging = LinearSegmentedColormap.from_list("blue_gray_red_cmap", colors)
            
            for j, slice_idx in enumerate(range(start_idx, end_idx)):
                # Plot predictions in the first row
              
                heatmap_x = axes1[0,j].pcolormesh(lon_grid, lat_grid, x_test[slice_idx], cmap='HomeyerRainbow', shading='auto',vmin=0, vmax=45)
                heatmap_x2 = axes1[1, j].pcolormesh(lon_grid, lat_grid, x_test[end_idx+slice_idx], cmap='HomeyerRainbow', shading='auto',vmin=0, vmax=45)
                heatmap_gt = axes1[2, j].pcolormesh(lon_grid, lat_grid, ground_truth[slice_idx], cmap='HomeyerRainbow', shading='auto',vmin=0, vmax=45)
                heatmap_gt2 = axes1[3, j].pcolormesh(lon_grid, lat_grid, ground_truth[end_idx+slice_idx], cmap='HomeyerRainbow', shading='auto',vmin=0, vmax=45)
                heatmap_pred = axes1[4, j].pcolormesh(lon_grid, lat_grid, predicted[slice_idx], cmap='HomeyerRainbow', shading='auto')
                heatmap_pred2 = axes1[5, j].pcolormesh(lon_grid, lat_grid, predicted[end_idx+slice_idx], cmap='HomeyerRainbow', shading='auto')
                norm11 = mcolors.TwoSlopeNorm(vmin=difference[slice_idx].min(), vcenter=0, vmax=difference[slice_idx].max())  # Adjust vmin and vmax as needed
                heatmap_diff = axes1[6,j].pcolormesh(lon_grid, lat_grid, difference[slice_idx], cmap=cmap_diverging, shading='auto', norm=norm11)
                norm12 = mcolors.TwoSlopeNorm(vmin=difference[end_idx+slice_idx].min(), vcenter=0, vmax=difference[end_idx+slice_idx].max())  # Adjust vmin and vmax as needed
                heatmap_diff2 = axes1[7,j].pcolormesh(lon_grid, lat_grid, difference[end_idx+slice_idx], cmap=cmap_diverging, shading='auto', norm=norm12)
                norm21 = mcolors.TwoSlopeNorm(vmin=difference_p[slice_idx].min(), vcenter=0, vmax=difference_p[slice_idx].max())  # Adjust vmin and vmax as needed
                heatmap_diff_p = axes1[8,j].pcolormesh(lon_grid, lat_grid, difference_p[slice_idx], cmap=cmap_diverging, shading='auto', norm=norm21)
                norm22 = mcolors.TwoSlopeNorm(vmin=difference_p[slice_idx].min(), vcenter=0, vmax=difference_p[slice_idx].max())  # Adjust vmin and vmax as needed
                heatmap_diff_p2 = axes1[9,j].pcolormesh(lon_grid, lat_grid, difference_p[end_idx+slice_idx], cmap=cmap_diverging, shading='auto', norm=norm22)
                axes1[0, j].set_title(f"X_Test {slice_idx}", fontsize=10, fontstyle='italic')
                axes1[0, j].axis("off")
                axes1[1, j].set_title(f"X_Test {end_idx+slice_idx}", fontsize=10, fontstyle='italic')
                axes1[1, j].axis("off")
                axes1[2, j].set_title(f"Y_Test {slice_idx}", fontsize = 10, fontstyle='italic')
                axes1[2, j].axis("off")
                axes1[3, j].set_title(f"Y_Test {slice_idx+end_idx}", fontsize=10, fontstyle='italic')
                axes1[3, j].axis("off")
                axes1[4, j].set_title(f"Y_Predicted {slice_idx}", fontsize = 10, fontstyle='italic')
                axes1[4, j].axis("off")
                axes1[5, j].set_title(f"Y_Predicted {end_idx+slice_idx}", fontsize=10, fontstyle='italic')
                axes1[5, j].axis("off")
                axes1[6, j].set_title(f"Diff_(gt,predicted) {slice_idx}", fontsize = 10, fontstyle='italic')
                axes1[6, j].axis("off")
                axes1[7, j].set_title(f"Diff_(gt,predicted) {slice_idx+end_idx}", fontsize=10, fontstyle='italic')
                axes1[7, j].axis("off")
                axes1[8, j].set_title(f"Diff_persistence {slice_idx}", fontsize = 10, fontstyle='italic')
                axes1[8, j].axis("off")
                axes1[9, j].set_title(f"Diff_persistence {slice_idx+end_idx}", fontsize=10, fontstyle='italic')
                axes1[9, j].axis("off")


        
        # Iterating over all the axes and adding coastline seperately.
            
            for row in range(10):
                for col in range(num_slices):
                    axes1[row, col].add_feature(cfeature.BORDERS, linestyle='-', alpha=1)
                    axes1[row, col].add_feature(cfeature.COASTLINE)
                    axes1[row, col].add_feature(cfeature.LAND, facecolor='none')

            

            data1 = [
                heatmap_x, 
                heatmap_x2, 
                heatmap_gt, 
                heatmap_gt2, 
                heatmap_pred, 
                heatmap_pred2, 
                heatmap_diff, 
                heatmap_diff2, 
                heatmap_diff_p,
                heatmap_diff_p2 
            ]

        
        
            for j in range(10):

                if j < len(data1):  # Prevent index out of range
                    cbar_pred = fig1.colorbar(data1[j], ax=axes1[j, :], orientation="horizontal", 
                                            fraction=0.1, pad=0.05, aspect=10, shrink=0.8)
                    cbar_pred.set_label("dBZ")

        


            # # Add coastlines and colorbar
            # ax.coastlines()
            # plt.colorbar(heatmap, ax=ax, label="Data Value")
            # plt.suptitle(f"Figure {end_idx/2}: Predictions (Row 1) vs Ground Truth (Row 2)")

            # # Create directory if it does not exist
            # save_path = os.path.join(dir, dir_name)
            # os.makedirs(save_path, exist_ok=True)  # `makedirs` ensures parent directories exist

            # Save the figure
            dir = "/home/vatsal/MOSDAC/predictions/"
            parent_path = os.path.dirname(dir)
            save_path1 = os.path.join(parent_path, f"NoNormalization_notcolorbar")
            if not os.path.exists(save_path1):  # Corrected syntax
                os.makedirs(save_path1, exist_ok=True)  # `makedirs` ensures parent directories exist
            
            folder_name = weight_add
            save_path2 = os.path.join(save_path1, folder_name)
            if not os.path.exists(save_path2):  # Corrected condition
                os.makedirs(save_path2, exist_ok=True)


            save_file = os.path.join(save_path2, f"Sample{i}.png")
            fig1.savefig(save_file)
            plt.close(fig1)
            print(f"Saved: {save_file}")
        

# lat_range = cal_lat_range("/home/vatsal/MOSDAC/RCTLS_01JUN2018_015137_L2B_STD.nc")
# lon_range = cal_lon_range("/home/vatsal/MOSDAC/RCTLS_01JUN2018_015137_L2B_STD.nc")
# # lat_range = 2
# # lon_range = 4
# types = [ "mse_", "mae_"]

# learning_rate = [0.001, 0.0001]

# activation = ["relu", "selu"]

# batch_size = [3,10, 20]

# root_dir = "/home/vatsal/MOSDAC/predictions/"
# for type in types:
#     for lr in learning_rate:
#         for ac in activation:
#             for bs in batch_size:
#                 dir_add = os.path.join(root_dir, f"{type}{lr}_{ac}_{bs}")
#                 if os.path.exists(dir_add):
#                     plot_slices3(dir_add,lat_range,lon_range,8,0,30)
#                 else:
#                     print("path doesnt exist")
#                     print(dir_add)
