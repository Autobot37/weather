import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader

from neuralop.models import FNO
from Utilities import *
from Rainy_Dataset import *
from Unet import UNet  # Not used in this code

# Load model weights
model_weights = ["FNO_Adam_0.005_2_64.pth"]
weights_dir = "/home/vatsal/MOSDAC/model_weights/FNO/"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = FNO(n_modes=(64, 64), hidden_channels=128, in_channels=10, out_channels=10)
model.to(device)

# Load latitude and longitude grids
lat_range = cal_lat_range("/home/vatsal/MOSDAC/RCTLS_05AUG2020_161736_L2B_STD.nc")
lon_range = cal_lon_range("/home/vatsal/MOSDAC/RCTLS_05AUG2020_161736_L2B_STD.nc")
lats_data = lat_range
lons_data = lon_range

# Load dataset
data_dir = '/home/vatsal/MOSDAC/train_test/full_dataset/'
train_dataset, test_dataset, val_dataset = rainy_dataset(data_dir, 10, 10)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Directory to save predictions
save_path = "/home/vatsal/Supreme/10_sequence/FNObest_diffcolorbar_try2/"
input_dir = os.path.join(save_path, "input")
result_dir = os.path.join(save_path, "result")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

for weight_file in model_weights:
    weight_path = os.path.join(weights_dir, weight_file)
    model.load_state_dict(torch.load(weight_path, weights_only=False))
    model.eval()

    for sample_idx, (x_test, y_test) in enumerate(test_loader):
        print(f"Sample: {sample_idx}")
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            pred = model(x_test)

        # Convert to NumPy
        x_test_np = x_test[0].cpu().numpy()
        y_test_np = y_test[0].cpu().numpy()
        pred_np = pred[0].cpu().numpy()

        # Clip predictions
        pred_np = np.clip(pred_np, 0, 60)

        # Get persistence (last input frame)
        persistence = x_test_np[-1]

        # Calculate differences
        difference = y_test_np - pred_np
        difference_p = y_test_np - persistence

        # Grid dimensions
        H, W = x_test_np.shape[-2:]
        lon_grid, lat_grid = np.meshgrid(lons_data, lats_data)
        lon_grid = lon_grid[:H, :W]
        lat_grid = lat_grid[:H, :W]

        # ======= Plot Input Sequence =======
        fig_input, axes_input = plt.subplots(1, 10, figsize=(50, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        for i in range(10):
            im = axes_input[i].pcolormesh(lon_grid, lat_grid, x_test_np[i], cmap='pyart_HomeyerRainbow',
                                          shading='auto', vmin=0, vmax=45)
            axes_input[i].set_title(f"Input {i}", fontsize=18, weight='bold')
            axes_input[i].axis("off")
            axes_input[i].add_feature(cfeature.BORDERS)
            axes_input[i].add_feature(cfeature.COASTLINE)
            axes_input[i].add_feature(cfeature.LAND, facecolor='none')

        fig_input.tight_layout(rect=[0, 0.07, 1, 0.95])  # leave space below

        # Create a new axis for the colorbar below the subplots
        cbar_ax = fig_input.add_axes([0.1, 0.02, 0.8, 0.02])  # [left, bottom, width, height]
        cbar = fig_input.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Reflectivity (dBZ)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        save_input_path = os.path.join(input_dir, f"Input_Sample{sample_idx}.png")
        fig_input.savefig(save_input_path, dpi=300, bbox_inches='tight')
        plt.close(fig_input)
        print(f"Saved: {save_input_path}")

        # ======= Plot GT, Pred, Diff =======
        fig_result, axes_result = plt.subplots(3, 10, figsize=(50, 25), subplot_kw={'projection': ccrs.PlateCarree()})
        for i in range(10):
            # GT
            im_gt = axes_result[0, i].pcolormesh(lon_grid, lat_grid, y_test_np[i], cmap='pyart_HomeyerRainbow',
                                                 shading='auto', vmin=0, vmax=45)
            axes_result[0, i].set_title(f"GT {i}", fontsize=18, weight='bold')
            axes_result[0, i].axis("off")

            # Predicted
            im_pred = axes_result[1, i].pcolormesh(lon_grid, lat_grid, pred_np[i], cmap='pyart_HomeyerRainbow',
                                                   shading='auto')
            axes_result[1, i].set_title(f"Pred {i}", fontsize=18, weight='bold')
            axes_result[1, i].axis("off")

            # Difference
            norm = mcolors.TwoSlopeNorm(vmin=difference[i].min(), vcenter=0, vmax=difference[i].max())
            im_diff = axes_result[2, i].pcolormesh(lon_grid, lat_grid, difference[i], cmap="seismic",
                                                   shading='auto', norm=norm)
            axes_result[2, i].set_title(f"Diff {i}", fontsize=18, weight='bold')
            axes_result[2, i].axis("off")

            # Common map features
            for row in range(3):
                axes_result[row, i].add_feature(cfeature.BORDERS)
                axes_result[row, i].add_feature(cfeature.COASTLINE)
                axes_result[row, i].add_feature(cfeature.LAND, facecolor='none')

                # Adjust layout to leave space below each row
        # Adjust layout to leave space below each row
        fig_result.tight_layout(rect=[0, 0.15, 1, 0.96])

        # ===== Thin colorbar under GT row =====
        cbar_ax1 = fig_result.add_axes([0.3, 0.665, 0.4, 0.01])  # [left, bottom, width, height]
        cbar_gt = fig_result.colorbar(im_gt, cax=cbar_ax1, orientation='horizontal', aspect= 30)
        # cbar_gt.set_label("Ground Truth dBZ", fontsize=14)
        cbar_gt.ax.tick_params(labelsize=12)

        # ===== Thin colorbar under Pred row =====
        cbar_ax2 = fig_result.add_axes([0.3, 0.435, 0.4, 0.01])
        cbar_pred = fig_result.colorbar(im_pred, cax=cbar_ax2, orientation='horizontal', aspect= 30)
        # cbar_pred.set_label("Predicted dBZ", fontsize=14)
        cbar_pred.ax.tick_params(labelsize=12)

        # ===== Thin colorbar under Diff row =====
        cbar_ax3 = fig_result.add_axes([0.3, 0.205, 0.4, 0.01])
        cbar_diff = fig_result.colorbar(im_diff, cax=cbar_ax3, orientation='horizontal', aspect= 30)
        # cbar_diff.set_label("Difference (GT - Pred)", fontsize=14)
        cbar_diff.ax.tick_params(labelsize=12)

        # ===== Save the figure =====
        save_result_path = os.path.join(result_dir, f"Result_Sample{sample_idx}.png")
        fig_result.savefig(save_result_path, dpi=300, bbox_inches='tight')
        plt.close(fig_result)
        print(f"Saved: {save_result_path}")

