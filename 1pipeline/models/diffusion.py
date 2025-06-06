import torch
import torch.nn as nn
import lightning as pl
from diffusers import UNet2DModel, DDIMScheduler
from torchmetrics.image import psnr, ssim
import lpips
import numpy as np

class DiffusionLit2D(pl.LightningModule):
    def __init__(
        self,
        image_size: int,
        in_window: int = 13,
        out_window: int = 12,
        lr: float = 5e-4,
        num_train_timesteps: int = 100,
        csi_threshold: float = 0.5,
    ):
        super().__init__()
        in_ch = in_window + out_window
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_ch,
            out_channels=out_window,
            center_input_sample=False,
            time_embedding_type="positional",
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            mid_block_type="UNetMidBlock2D",
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(224, 448, 672, 896),
            layers_per_block=2,
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type="conv",
            upsample_type="conv",
            dropout=0.0,
            act_fn="silu",
            attention_head_dim=8,
            norm_num_groups=32,
            attn_norm_num_groups=None,
            norm_eps=1e-5,
            resnet_time_scale_shift="default",
            add_attention=True,
            class_embed_type=None,
            num_class_embeds=None,
            num_train_timesteps=num_train_timesteps,
        )
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
        self.loss_fn = nn.MSELoss()
        self.lpips_fn = lpips.LPIPS(net="alex")
        self.lr = lr
        self.in_w = in_window
        self.out_w = out_window
        self.csi_th = csi_threshold

    def forward(self, noisy, t, cond):
        x = torch.cat((noisy, cond), dim=1)
        return self.unet(x, t).sample

    def training_step(self, batch, batch_idx):
        v = batch["vil"].permute(0, 3, 1, 2).float()    # [B,25,H,W]
        cond = v[:, : self.in_w, :, :]                  # [B,13,H,W]
        tgt = v[:, self.in_w :, :, :]                   # [B,12,H,W]
        noise = torch.randn_like(tgt)
        T = torch.randint(0, self.scheduler.num_train_timesteps, (tgt.shape[0],), device=self.device).long()
        noisy = self.scheduler.add_noise(tgt, noise, T)
        pred = self.forward(noisy, T, cond)
        loss = self.loss_fn(pred, noise)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        v = batch["vil"].permute(0, 3, 1, 2).float()
        cond = v[:, : self.in_w, :, :]
        tgt = v[:, self.in_w :, :, :]
        B = tgt.shape[0]
        gen = torch.randn(B, self.out_w, v.shape[2], v.shape[3], device=self.device)
        self.scheduler.set_timesteps(self.scheduler.num_train_timesteps)
        for t in self.scheduler.timesteps:
            tb = torch.full((B,), t, device=self.device, dtype=torch.long)
            p = self.unet(torch.cat((gen, cond), dim=1), tb).sample
            gen = self.scheduler.step(p, t, gen).prev_sample
        gen = gen.clamp(0, 1)
        tgt_cl = tgt.clamp(0, 1)
        lp_v, ps_v, ss_v, cs_v = [], [], [], []
        self.lpips_fn = self.lpips_fn.to(self.device)
        for i in range(B):
            p_i = gen[i : i + 1]   # [1,12,H,W]
            r_i = tgt_cl[i : i + 1]
            lp_c, ps_c, ss_c = [], [], []
            for c in range(self.out_w):
                p_s = p_i[:, c : c + 1, :, :]         # [1,1,H,W]
                r_s = r_i[:, c : c + 1, :, :]
                p3 = p_s.repeat(1, 3, 1, 1) * 2 - 1
                r3 = r_s.repeat(1, 3, 1, 1) * 2 - 1
                lp_c.append(self.lpips_fn(p3, r3).item())
                ps_c.append(psnr(p_s, r_s, data_range=1.0).item())
                ss_c.append(ssim(p_s, r_s, data_range=1.0).item())
            lp = np.mean(lp_c); ps = np.mean(ps_c); ss = np.mean(ss_c)
            pred_bin = (p_i >= self.csi_th); tgt_bin = (r_i >= self.csi_th)
            TP = (pred_bin & tgt_bin).sum().item()
            FP = (pred_bin & ~tgt_bin).sum().item()
            FN = (~pred_bin & tgt_bin).sum().item()
            cs = TP / (TP + FP + FN + 1e-6)
            lp_v.append(lp); ps_v.append(ps); ss_v.append(ss); cs_v.append(cs)
        metrics = {
            "val/LPIPS": np.mean(lp_v),
            "val/PSNR": np.mean(ps_v),
            "val/SSIM": np.mean(ss_v),
            "val/CSI": np.mean(cs_v),
        }
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
image_size = 384
in_window = 13
out_window = 12
model2d = DiffusionLit2D(image_size=image_size, in_window=in_window, out_window=out_window)

from datasets.dataset_sevir import SEVIRLightningDataModule
dm = SEVIRLightningDataModule()
dm.prepare_data(); dm.setup()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
for sample in train_loader:
    shape = sample["vil"].shape
    print(f"Training sample shape: {shape}")
    assert shape[3] == in_window + out_window, f"Expected input shape to have {in_window + out_window} channels, got {shape[1]}"
    break

trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=[0],
    log_every_n_steps=10,
    limit_train_batches = 2, limit_val_batches = 2
)
trainer.fit(model2d, train_dataloaders=train_loader, val_dataloaders=val_loader)
