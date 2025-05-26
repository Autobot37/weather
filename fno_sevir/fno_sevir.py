from neuralop.models import FNO
from fno_dataset import *
import wandb
from torchmetrics.functional import structural_similarity_index_measure as ssim
from Prediction_metrics.Calculate_pred_met import *
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Run the extreme rainfall event detection model.")
    
    # Dataset parameters
    parser.add_argument('--sequence_length', type=int, default=20, help='Number of sequential files per data point.')
    
    # Model parameters
    parser.add_argument('--input_channel', type=int, default=10, help='Number of channels to pass as the input')
    parser.add_argument('--output_channel', type=int, default=10, help='Number of channels in the output')
    parser.add_argument('--optimizer', type=str, default="SGD", help='optimizer.')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer.')
    
  
    # parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    
    return parser.parse_args()

args = parse_args()

# Use args.sequence_length consistently
train_loader = SEVIRDataLoader(seq_len=args.sequence_length, verbose=True, batch_size=args.batch_size, layout='NTHW', year_filter=['2017', '2018'], split= "train")
val_loader = SEVIRDataLoader(seq_len=args.sequence_length, verbose=True, batch_size=args.batch_size, layout='NTHW', year_filter='2019', split= "val")
test_loader = SEVIRDataLoader(seq_len=args.sequence_length, verbose=True, batch_size=args.batch_size, layout='NTHW', year_filter='2019', split= "test")

for loader in [train_loader, val_loader, test_loader]:
    for sample in loader:
        vil = sample['vil']
        print(f"Sample shape: {vil.shape}, dtype: {vil.dtype}, device: {vil.device}")
        break

print(f"Train samples: {len(train_loader._samples)}")
print(f"Val samples: {len(val_loader._samples)}")
print(f"Test samples: {len(test_loader._samples)}")

os.makedirs('fno_plots', exist_ok=True)

model = FNO(n_modes=(64, 64), hidden_channels=128,
               in_channels=args.input_channel, out_channels=args.output_channel)
model = model.to('cuda')
torch.compile(model)

criterion = nn.MSELoss()

if args.optimizer.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer.lower() == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

vil_colormap, vil_norm = vil_cmap(encoded=True)

wandb.init(project= "FNO_Savir")
best_val_loss = float('inf')

print('Training')
for epoch in range(args.epochs):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs} [Train]')

    avg_train_loss = []
    for batch_idx, data in pbar:
        vil_sequence = data['vil']
        vil_sequence = vil_sequence.to('cuda')
        
        input_frames = vil_sequence[:, :10]
        target_frames = vil_sequence[:, 10:]
        
        predicted_frames = model(input_frames)
        loss = criterion(predicted_frames, target_frames)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_train_loss.append(loss.item())
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    train_losses = np.mean(avg_train_loss)

    model.eval()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch+1}/{args.epochs} [Validation]')

    avg_val_loss = []
    sample_metrics_pcc = []
    sample_metrics_ssim = []
    sample_metrics_psnr = []
    sample_metrics_acc = []
    sample_metrics_rmse = []

    plot_pred = None
    plot_target = None

    with torch.no_grad():
        for batch_idx, data in pbar:
            vil_sequence = data['vil']
            vil_sequence = vil_sequence.to('cuda')
            
            input_frames = vil_sequence[:, :args.sequence_length//2]
            y_val = vil_sequence[:, args.sequence_length//2:]
            
            val_pred = model(input_frames)
            val_loss = criterion(val_pred, y_val)
            
            avg_val_loss.append(val_loss.item())
            
            # Fix sequence length parameter - should be output sequence length
            output_seq_len = args.sequence_length//2
            sample_metrics_pcc.append(cal_batch_avg_pcc(y_val, val_pred, output_seq_len))
            sample_metrics_ssim.append(ssim(y_val, val_pred, data_range = 50.0).cpu().numpy())
            sample_metrics_psnr.append(cal_batch_avg_psnr(y_val, val_pred, output_seq_len))
            sample_metrics_acc.append(compute_batch_avg_acc(y_val, val_pred))
            sample_metrics_rmse.append(cal_batch_avg_rmse(y_val, val_pred, output_seq_len))
            pbar.set_postfix({'Loss': f'{val_loss.item():.4f}'})
            
            if batch_idx == 0:
                plot_pred = val_pred
                plot_target = y_val

    val_losses = np.mean(avg_val_loss)

    sample_metrics_pcc = np.array(sample_metrics_pcc)

    sample_pcc = np.mean(sample_metrics_pcc)
    
    sample_ssim = np.mean(sample_metrics_ssim)

    sample_psnr = np.mean(sample_metrics_psnr)

    sample_acc = np.mean(sample_metrics_acc)

    sample_rmse = np.mean(sample_metrics_rmse)

    if not os.path.exists('/home/vatsal/NWM/fno_sevir/model_weights/FNO/'):
        os.makedirs('/home/vatsal/NWM/fno_sevir/model_weights/FNO/', exist_ok=True)

    if val_losses < best_val_loss:
        best_val_loss = val_losses
        torch.save(model.state_dict(), f'/home/vatsal/NWM/fno_sevir/model_weights/FNO/weights.pt')  # save best checkpoint
        print(f"Saved best model at epoch {epoch} with val_loss: {val_losses:.4f}")
        print(f"Model saved, val_loss: {val_losses}")

    if epoch % 1 == 0 and plot_pred is not None:
        try:
            # Determine number of frames to plot
            num_frames = min(10, plot_pred.shape[1])
            fig, axes = plt.subplots(2, num_frames, figsize=(2*num_frames, 6))
            
            pred_np = plot_pred[0].detach().cpu().numpy() * 255
            target_np = plot_target[0].detach().cpu().numpy() * 255

            for i in range(num_frames):
                if num_frames == 1:
                    axes[0].imshow(target_np[i], cmap=vil_colormap, norm=vil_norm)
                    axes[0].set_title(f'Target Frame {i+1}')
                    axes[0].axis('off')

                    axes[1].imshow(pred_np[i], cmap=vil_colormap, norm=vil_norm)
                    axes[1].set_title(f'Predicted Frame {i+1}')
                    axes[1].axis('off')
                else:
                    axes[0, i].imshow(target_np[i], cmap=vil_colormap, norm=vil_norm)
                    axes[0, i].set_title(f'Target Frame {i+1}')
                    axes[0, i].axis('off')

                    axes[1, i].imshow(pred_np[i], cmap=vil_colormap, norm=vil_norm)
                    axes[1, i].set_title(f'Predicted Frame {i+1}')
                    axes[1, i].axis('off')

            plt.tight_layout()
            
            wandb.log({
                "Plot": wandb.Image(fig),
                "Train Loss": train_losses,
                "Validation_loss": val_losses,
                "PCC": sample_pcc,
                "SSIM": sample_ssim,
                "PSNR": sample_psnr,
                "ACC": sample_acc,
                "RMSE": sample_rmse
            })

            plt.close(fig)
        except Exception as e:
            print(f"Error creating plot: {e}")
        
    print(f"Epoch {epoch+1}, Train Loss: {train_losses:.6f}, Val Loss: {val_losses:.6f}, SSIM: {sample_ssim}, PSNR: {sample_psnr}, ACC: {sample_acc}, RMSE :{sample_rmse} ")

wandb.finish()