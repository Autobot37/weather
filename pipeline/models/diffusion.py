import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from diffusers import UNet2DModel, DDIMScheduler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from einops import rearrange
from tqdm import tqdm
from metrics import *
# -----------------------------
# 1. Hyperparameters & Setup
# -----------------------------
image_size = 128
in_window = 4
out_window = 4
lr = 5e-4
num_train_timesteps = 100
csi_threshold = 0.5
num_epochs = 1
max_val_batches = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Diffusion Model & Scheduler
# -----------------------------
class DiffusionNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = in_window + out_window
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_ch,
            out_channels=out_window,
            center_input_sample=False,
            time_embedding_type="positional",
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            mid_block_type="UNetMidBlock2D",
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(32, 32, 64),
            layers_per_block=1,
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type="conv",
            upsample_type="conv",
            dropout=0.0,
            act_fn="silu",
            attention_head_dim=4,
            norm_num_groups=16,
            norm_eps=1e-5,
            resnet_time_scale_shift="default",
            num_train_timesteps=num_train_timesteps,
        )
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        x = torch.cat((noisy, cond), dim=1)
        return self.unet(x, timesteps).sample

model = DiffusionNet2D().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# -----------------------------
# 4. DataModule & Loaders
# -----------------------------
from datasets.dataset_sevir import SEVIRLightningDataModule

dm = SEVIRLightningDataModule()
dm.prepare_data()
dm.setup()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
for loader in [train_loader, val_loader]:
    for sample in loader:
        data = sample['vil']
        print("sample shape:", data.shape)
        print("min/max:", data.min().item(), data.max().item())
        break

# -----------------------------
# 5. CSV Logger
# -----------------------------
csv_file = f"metrics_log_{int(time.time())}.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 'train_loss',
        'LPIPS','PSNR','SSIM','POD','SUCR','CSI','BIAS','HSS'
    ])

# -----------------------------
# 6. Train & Validate
# -----------------------------
for epoch in range(1, num_epochs+1):
    # Training
    model.train()
    train_losses = []
    tbar = tqdm(train_loader, desc=f'Epoch {epoch} Train', leave=False)
    for batch in tbar:
        vil = batch['vil'].permute(0,3,1,2).float().to(device)
        vil0 = vil[:,0:1]; vil = vil - vil0
        cond, tgt = vil[:,:in_window], vil[:,in_window:]
        noise = torch.randn_like(tgt)
        B = tgt.size(0)
        T = torch.randint(0, num_train_timesteps, (B,), device=device)
        noisy = model.scheduler.add_noise(tgt, noise, T)
        pred = model(noisy, T, cond)
        loss = mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        tbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_train = np.mean(train_losses)
    import time
    t = time.time()
    torch.save(model.state_dict(), f"unet_diff_epoch_{epoch}_{time.time()}.pth")
    # Validation
    model.eval()
    vals = {'LPIPS':[], 'PSNR':[], 'SSIM':[], 'POD':[], 'SUCR':[], 'CSI':[], 'BIAS':[], 'HSS':[]}
    vbar = tqdm(val_loader, desc=f'Epoch {epoch} Val', leave=False, )
    with torch.no_grad():
        for idx, batch in enumerate(vbar):
            if idx >= max_val_batches:
                break
            vil = batch['vil'].permute(0,3,1,2).float().to(device)
            vil0 = vil[:,0:1]; vil = vil - vil0
            cond, tgt = vil[:,:in_window], vil[:,in_window:]
            B = tgt.size(0)
            # generate
            gen = torch.randn(B, out_window, image_size, image_size, device=device)
            model.scheduler.set_timesteps(num_train_timesteps)
            for t in model.scheduler.timesteps:
                tb = torch.full((B,), t, device=device, dtype=torch.long)
                eps = model(gen, tb, cond)
                gen = model.scheduler.step(eps, t, gen).prev_sample
            # clip/rescale
            gen = gen.clamp(-1,1).add(1).div(2)
            tgt = tgt.clamp(-1,1).add(1).div(2)

            for i in range(B):
                p = gen[i:i+1]; r = tgt[i:i+1]
                lp = 0
                ps = psnr_fn(p, r).item()
                ss = ssim_fn(p, r).item()
                h,m,f,cor = calc_hits_misses_fas(p, r, csi_threshold)
                vals['POD'].append(pod(h,m,f).item())
                vals['SUCR'].append(sucr(h,m,f).item())
                vals['CSI'].append(csi(h,m,f).item())
                vals['BIAS'].append(bias(h,m,f).item())
                vals['HSS'].append(hss(h,m,f,cor).item())
                vals['LPIPS'].append(lp)
                vals['PSNR'].append(ps)
                vals['SSIM'].append(ss)
            vbar.set_postfix({k: f"{np.mean(vals[k]):.4f}" for k in vals})

    # write CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{avg_train:.6f}",
            f"{np.mean(vals['LPIPS']):.6f}",
            f"{np.mean(vals['PSNR']):.6f}",
            f"{np.mean(vals['SSIM']):.6f}",
            f"{np.mean(vals['POD']):.6f}",
            f"{np.mean(vals['SUCR']):.6f}",
            f"{np.mean(vals['CSI']):.6f}",
            f"{np.mean(vals['BIAS']):.6f}",
            f"{np.mean(vals['HSS']):.6f}",
        ])

    print(f"Epoch {epoch} | Train: {avg_train:.4f} | LPIPS: {np.mean(vals['LPIPS']):.4f}, PSNR: {np.mean(vals['PSNR']):.4f}, SSIM: {np.mean(vals['SSIM']):.4f}, CSI: {np.mean(vals['CSI']):.4f}")

print("Done.")
