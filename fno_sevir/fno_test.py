import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import argparse 
from neuralop.models import FNO
from fno_dataset import *
from datetime import datetime
from torchmetrics.functional import structural_similarity_index_measure as ssim
from metrics import calculate_metrics_dict
import pandas as pd

def plot(predicted_frames, target_frames , vil_colormap, vil_norm, epoch, batch_idx=0):
    num_frames = predicted_frames.shape[1]
    fig, axes = plt.subplots(2, num_frames, figsize=(2*num_frames, 6))
    fig.suptitle(f'Epoch {epoch+1} - Predicted vs Target Frames', fontsize=16)
    # Move to cpu and scale to 0-255 for visualization
    pred_np = predicted_frames[0].detach().cpu().numpy() * 255
    target_np = target_frames[0].detach().cpu().numpy() * 255
    pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
    target_np = np.clip(target_np, 0, 255).astype(np.uint8)
    
    for i in range(num_frames):
        axes[0, i].imshow(target_np[i], cmap=vil_colormap, norm=vil_norm)
        axes[0, i].set_title(f'Target Frame {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(pred_np[i], cmap=vil_colormap, norm=vil_norm)
        axes[1, i].set_title(f'Predicted Frame {i+1}')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f'fno_test_plots/epoch_{epoch+1}_batch_{batch_idx}.png', dpi = 300)
    # plt.show()'
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the extreme rainfall event detection model.")
    parser.add_argument('--sequence_length', type=int, default=20, help='Number of sequential files per data point.')
    parser.add_argument('--input_channel', type=int, default=10, help='Number of channels to pass as the input')
    parser.add_argument('--output_channel', type=int, default=10, help='Number of channels in the output')
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer.')
    return parser.parse_args()

args = parse_args()
weights_path = '/home/vatsal/NWM/fno_sevir/model_weights/FNO/'
if not os.path.exists(weights_path):
    os.makedirs(weights_path, exist_ok=True)
os.makedirs('fno_test_plots', exist_ok=True)

test_loader = SEVIRDataLoader(start_date=datetime(2019, 5, 1), end_date=datetime(2019, 7, 1), batch_size=1, shuffle=False, seq_len=20, stride = 12, rescale_method='01', preprocess=True)
#test-> [2019, 5, 1] to [2019, 7, 1]
print("Train dataset size:", len(test_loader))
for loader in [test_loader]:
    for sample in loader:
        vil = sample['vil']
        print(f"Sample shape: {vil.shape}, dtype: {vil.dtype}. maximum value: {vil.max().item()}, minimum testue: {vil.min().item()}")
        break

model = FNO(n_modes=(64, 64), hidden_channels=128,
               in_channels=args.input_channel, out_channels=args.output_channel)
model.load_state_dict(torch.load(os.path.join(weights_path, "weights.pth"), weights_only=False))
model.eval()
model = model.to('cuda')

plot_idxs = list(range(0, 1500, 50))

vil_colormap, vil_norm = vil_cmap(encoded=True)
all_rows = []

with torch.no_grad():
    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, data in test_pbar:
        if batch_idx < 10:
            print("Processing batch:", batch_idx)
        vil_sequence = data['vil']
        vil_sequence = vil_sequence.to('cuda')
        vil_sequence = vil_sequence.permute(0, 3, 1, 2)
        assert vil_sequence.shape[1] == 20, f"Expected 20 channels, got {vil_sequence.shape[1]}"
        input_frames = vil_sequence[:, :10]
        target_frames = vil_sequence[:, 10:]
        
        predicted_frames = model(input_frames)
        if batch_idx == 0:
            print(f"Input frames shape: {input_frames.shape}, Target frames shape: {target_frames.shape}, Predicted frames shape: {predicted_frames.shape}")

        # if batch_idx in plot_idxs:
        #     plot(predicted_frames, target_frames, vil_colormap, vil_norm, 1, batch_idx)

        input_frames = input_frames.cpu().numpy()
        target_frames = target_frames.cpu().numpy()
        predicted_frames = predicted_frames.cpu().numpy()
        
        metrics_dict = calculate_metrics_dict(predicted_frames, target_frames, input_frames)
        row_df = pd.DataFrame([metrics_dict])  # wrap dict in list → 1-row DataFrame
        all_rows.append(row_df)
    test_loader.reset()

final_df = pd.concat(all_rows, ignore_index=True)
final_df.to_csv('fno_test_metrics.csv', index=False)