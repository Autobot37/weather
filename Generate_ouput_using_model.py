import torch
import pickle
from neuralop.models.fno import FNO
from Utilities.Rainy_Dataset import *
from Prediction_metrics.Calculate_pred_met import *
import argparse
from torchmetrics.functional import structural_similarity_index_measure as ssim
from DiffCast.models.diffcast import *
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


device = torch.device("cuda:0")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
#  All addresses

dir_add = "/home/vatsal/MOSDAC/Model_weights/"
# model_weights = os.listdir(dir_add)
model_weights = ['Diffcast_2 epochs.pt']

save_path = "/home/vatsal/MOSDAC/Model_plots/"
input_dir = os.path.join(save_path, "input")
result_dir = os.path.join(save_path, "result")

data_dir = '/home/vatsal/MOSDAC/train_test/full_dataset/'
train_dataset, test_dataset, val_dataset = rainy_dataset(data_dir,10,10)
test_loader = create_loader(test_dataset, batch_size=1)

backbone_net = get_FNO_model(n_modes=64, hidden_channels=64, in_channels=10, out_channels=10)
backbone_net.to(device)

model = get_model(
        img_channels=1,
        dim=128,
        dim_mults=(2, 4, 8, 16),
        T_in=10,
        T_out=10,
    ).to(device)


model.load_backbone(backbone_net)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

x_test_lis = []
y_test_lis = []
y_pred_lis = []


lat_range = np.load('/home/vatsal/MOSDAC/Lat_lon_ranges/lat_range.npy')
lon_range = np.load('/home/vatsal/MOSDAC/Lat_lon_ranges/lon_range.npy')


os.makedirs(input_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)


for weight_add in model_weights:
    path = os.path.join(dir_add, weight_add)
    state_dict = torch.load(path, weights_only=False)
    del state_dict['_metadata']
    
    model.load_state_dict(state_dict)

    model.eval()
   

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    x_test_lis = []
    y_test_lis = []
    y_pred_lis = []

   
    for sample_idx, (x_test, y_test) in enumerate(test_loader):
        
        x_test, y_test = x_test.to(device), y_test.to(device)

        # [B, T, H, W] → [B, T, 1, H, W]
        x_test = x_test.unsqueeze(2)
        y_test = y_test.unsqueeze(2) 

        with torch.no_grad():
            predictions, _ = model.predict(x_test, y_test, compute_loss=False)

        # Optional: Remove channel dim for visualization
        x_test = x_test.squeeze(2)
        y_test = y_test.squeeze(2)
        predictions = predictions.squeeze(2)

        x_test_np = x_test[0].cpu().numpy()
        y_test_np = y_test[0].cpu().numpy()
        pred_np = predictions[0].cpu().numpy()


        persistence = x_test_np[-1]

        # Calculate differences
        difference = y_test_np - pred_np
        difference_p = y_test_np - persistence



        # ======= Plot Input Sequence =======
        
        
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
        
        lon_grid = lon_grid[112:368, 112:368]
        lat_grid = lat_grid[112:368, 112:368]
        

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
                                                   shading='auto', vmin=0, vmax=2)
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


        # Move prediction back to CPU and convert to numpy

        
    #     x_test_lis.append(x_test)
    #     y_test_lis.append(y_test)
    #     y_pred_lis.append(pred)


    # model_name1 = weight_add.split(".")[0]
    # model_name2 = weight_add.split(".")[1]
    
    # plot_predictionmetric_scores2(os.path.join("/home/vatsal/MOSDAC/predictions2/Conv_LSTM", model_name1 + model_name2), x_test_lis, y_test_lis, y_pred_lis, len(x_test_lis), 16)
