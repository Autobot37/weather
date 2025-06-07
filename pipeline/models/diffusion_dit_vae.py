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
from omegaconf import OmegaConf
from dit import *
from CasCast.networks.prediff.taming.autoencoder_kl import AutoencoderKL
from collections import OrderedDict

"""
Full resolution is more important.

Cascast -> transformer(VT), (T * H * W)**2 
Diffcast -> UNet3D [B, T, C, H, W]
            EF  (T H/P W/P)
            
frames = [T, H, W]
frames -= frames0

cond = frames[:inwindow]  # [B, in_window, C, H, W]
tgt = frames[in_window:]  # [B, out_window, C, H, W]


frames = [0, 1]
frames0 = [0, 1]
frames = [-1, 1]
target = [-1, 1]

"""
def load_checkpoint(checkpoint_path, model):
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint_dict['model']
    ckpt_submodels = list(checkpoint_model.keys())
    print(ckpt_submodels)
    submodels = ['autoencoder_kl']
    key = 'autoencoder_kl'
    if key not in submodels:
        print(f"warning!!!!!!!!!!!!!: skip load of {key}")
    new_state_dict = OrderedDict()
    for k, v in checkpoint_model[key].items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("net."):
            name = name[len("net."):]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    from termcolor import colored
    print(colored(f"loaded {key} successfully the game is on", 'green'))
    return model
"""
unet2d hf same as diffcast
cascast used dit 
options 
unet no need for latent here.
transformer -> dit one, earthformer 
"""
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
        path = "/home/vatsal/NWM/pipeline/configs/models/dit_encoded.yaml"
        config = OmegaConf.load(path)
        casformer = CasFormer(arch='DiT-custom', config=config)
        self.model = casformer
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        return self.model(noisy, timesteps, cond)

model = DiffusionNet2D().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# -----------------------------
# vae
# -----------------------------
config_path = "/home/vatsal/NWM/pipeline/configs/models/autoencoder.yaml"
config = OmegaConf.load(config_path)
autoencoder = AutoencoderKL(**config).eval()
autoencoder = load_checkpoint("/home/vatsal/NWM/autoencoder_ckpt.pth", autoencoder)
autoencoder = autoencoder.to(device)
for param in autoencoder.parameters():
    param.requires_grad = False

@torch.no_grad()
def encode(x):
    ##[B, T, C, H, W] 
    assert x.shape[1] == in_window + out_window, "Input shape must match in_window + out_window"
    assert x.shape[2] == 1, "Input shape must have channel dimension of 1"
    stk = []
    for i in range(x.shape[1]):
        frame = x[:, i:i+1, :, :, :].squeeze(1)  # [B, C, H, W]
        frame = autoencoder.encode().sample()
        frame = frame.unsqueeze(1)  # [B, 1, C, H, W]
        stk.append(frame)
    return torch.cat(stk, dim=1)  # [B, T, C, H, W]

@torch.no_grad()
def decode(x):
    channels = x.shape[2]
    assert channels == 4, "Input shape must have channel dimension of 4"
    stk = []
    for i in range(x.shape[1]):
        frame = x[:, i:i+1, :, :, :].squeeze(1)  # [B, C, H, W]
        frame = autoencoder.decode(frame)
        frame = frame.unsqueeze(1)  # [B, 1, C, H, W]
        stk.append(frame)
    return torch.cat(stk, dim=1)  # [B, T, C, H, W]

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
        vil = batch['vil'].permute(0,3,1,2).unsqueeze(2).float().to(device)
        vil = encode(vil)  # B T C H W
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
    torch.save(model.state_dict(), f"dit_diff_epoch_{epoch}_{time.time()}.pth")
    # Validation
    model.eval()
    vals = {'LPIPS':[], 'PSNR':[], 'SSIM':[], 'POD':[], 'SUCR':[], 'CSI':[], 'BIAS':[], 'HSS':[]}
    vbar = tqdm(val_loader, desc=f'Epoch {epoch} Val', leave=False, )
    with torch.no_grad():
        for idx, batch in enumerate(vbar):
            if idx >= max_val_batches:
                break
            vil = batch['vil'].permute(0,3,1,2).unsqueeze(2).float().to(device)
            vil = encode(vil)  # B T C H W
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
            gen = decode(gen)  # B T C H W
            tgt = decode(tgt)  # B T C H W

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
