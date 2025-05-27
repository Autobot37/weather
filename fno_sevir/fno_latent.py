import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import argparse 
import wandb
from neuralop.models import FNO
from fno_dataset import *
from datetime import datetime
from torchmetrics.functional import structural_similarity_index_measure as ssim
from Prediction_metrics.Calculate_pred_met import *

wandb.init(project= "FNO-SEVIR")

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
    wandb.log({
        f"Plot_Epoch_{epoch}_Batch_{batch_idx}": wandb.Image(fig),
    })
    plt.show()

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

train_loader = SEVIRDataLoader(start_date=datetime(2017, 1, 1), end_date=datetime(2019, 1, 1), batch_size=4, shuffle=True, seq_len=20, stride = 20, rescale_method='01', preprocess=True)
val_loader = SEVIRDataLoader(start_date=datetime(2019, 5, 1), end_date=datetime(2019, 7, 1), batch_size=4, shuffle=False, seq_len=20, stride = 20, rescale_method='01', preprocess=True)
#test-> [2019, 5, 1] to [2019, 7, 1]
print("Train dataset size:", len(train_loader))
print("Validation dataset size:", len(val_loader))
for loader in [train_loader, val_loader]:
    for sample in loader:
        vil = sample['vil']
        print(f"Sample shape: {vil.shape}, dtype: {vil.dtype}. maximum value: {vil.max().item()}, minimum value: {vil.min().item()}")
        break

model = FNO(n_modes=(64, 64), hidden_channels=128,
               in_channels=args.input_channel, out_channels=args.output_channel)
model = model.to('cuda')

criterion = nn.MSELoss()

if args.optimizer.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer.lower() == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

best_val_loss = float('inf')

num_epochs = args.epochs

vil_colormap, vil_norm = vil_cmap(encoded=True)

for epoch in range(num_epochs):
    model.train()
    avg_train_losses = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    for batch_idx, data in pbar:
        vil_sequence = data['vil']
        vil_sequence = vil_sequence.to('cuda')
        vil_sequence = vil_sequence.permute(0, 3, 1, 2)
        assert vil_sequence.shape[1] == 20, f"Expected 10 channels, got {vil_sequence.shape[1]}"
        input_frames = vil_sequence[:, :10]
        target_frames = vil_sequence[:, 10:]
        
        predicted_frames = model(input_frames)
        loss = criterion(predicted_frames, target_frames)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        avg_train_losses.append(loss.item())
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    train_loader.reset()
    model.eval()

    avg_val_losses = []
    sample_metrics_pcc = []
    sample_metrics_ssim = []
    sample_metrics_psnr = []
    sample_metrics_acc = []
    sample_metrics_rmse = []
    
    with torch.no_grad():
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
        for batch_idx, data in val_pbar:
            vil_sequence = data['vil']
            vil_sequence = vil_sequence.to('cuda')
            vil_sequence = vil_sequence.permute(0, 3, 1, 2)
            assert vil_sequence.shape[1] == 20, f"Expected 20 channels, got {vil_sequence.shape[1]}"
            input_frames = vil_sequence[:, :10]
            target_frames = vil_sequence[:, 10:]
            
            predicted_frames = model(input_frames)
            loss = criterion(predicted_frames, target_frames)
            
            avg_val_losses.append(loss.item())
            val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Convert to numpy for metrics - keep tensors on CPU for metrics
            y_val = target_frames.detach().cpu()
            val_pred = predicted_frames.detach().cpu()

            output_seq_len = args.output_channel  # or sequence_length//2
            sample_metrics_pcc.append(cal_batch_avg_pcc(y_val, val_pred, output_seq_len))
            sample_metrics_ssim.append(ssim(y_val, val_pred, data_range=1.0).cpu().numpy())  # data is normalized to [0,1]
            sample_metrics_psnr.append(cal_batch_avg_psnr(y_val, val_pred, output_seq_len))
            sample_metrics_acc.append(compute_batch_avg_acc(y_val, val_pred))
            sample_metrics_rmse.append(cal_batch_avg_rmse(y_val, val_pred, output_seq_len))

            if batch_idx % 100 == 0:
                plot(predicted_frames, target_frames, vil_colormap, vil_norm, epoch, batch_idx)

    val_loader.reset()

    train_loss = np.mean(avg_train_losses)
    val_loss = np.mean(avg_val_losses)
    sample_metrics_pcc = np.array(sample_metrics_pcc)
    sample_pcc = np.mean(sample_metrics_pcc)
    sample_ssim = np.mean(sample_metrics_ssim)
    sample_psnr = np.mean(sample_metrics_psnr)
    sample_acc = np.mean(sample_metrics_acc)
    sample_rmse = np.mean(sample_metrics_rmse)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(weights_path, "weights.pth"))
        print(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")

    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Validation_loss": val_loss,
        "PCC": sample_pcc,
        "SSIM": sample_ssim,
        "PSNR": sample_psnr,
        "ACC": sample_acc,
        "RMSE": sample_rmse
    })

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, SSIM: {sample_ssim}, PSNR: {sample_psnr}, ACC: {sample_acc}, RMSE :{sample_rmse} ")

wandb.finish()