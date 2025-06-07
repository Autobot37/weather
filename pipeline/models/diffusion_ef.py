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
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
"""
unet2d hf same as diffcast
cascast used dit 
options 
unet no need for latent here.
transformer -> dit one, earthformer [[batch, time, height, width, channels]]
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Diffusion Model & Scheduler
# -----------------------------
class DiffusionNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        path = "/home/vatsal/NWM/pipeline/configs/models/earthformer.yaml"
        model_cfg = OmegaConf.load(path)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
        self.model = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        inp = torch.concat([noisy, cond], dim=1)  # [B, C+C, H, W]
        return self.model(inp)

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
        vil = batch['vil'].permute(0, 3, 1, 2).unsqueeze(4).float().to(device)
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
    torch.save(model.state_dict(), f"ef_diff_epoch_{epoch}_{time.time()}.pth")
    # Validation
    model.eval()
    vals = {'LPIPS':[], 'PSNR':[], 'SSIM':[], 'POD':[], 'SUCR':[], 'CSI':[], 'BIAS':[], 'HSS':[]}
    vbar = tqdm(val_loader, desc=f'Epoch {epoch} Val', leave=False, )
    with torch.no_grad():
        for idx, batch in enumerate(vbar):
            if idx >= max_val_batches:
                break
            vil = batch['vil'].permute(0, 3, 1, 2).unsqueeze(4).float().to(device)
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
