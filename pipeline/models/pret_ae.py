import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
from pipeline.utils import calc_metrics
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from CasCast.networks.prediff.taming.autoencoder_kl import AutoencoderKL
from pipeline.utils import load_checkpoint_cascast
from pytorch_lightning.loggers import WandbLogger
os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'

"""
384x384
"""
##
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=4, bottleneck_channels=16):
        super().__init__()
        # conv reduce channels then downsample 3×
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.LayerNorm([bottleneck_channels, 48, 48]),  # LayerNorm for stability
            nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([bottleneck_channels, 24, 24]),
            nn.ReLU()
        )  # 48→24
        self.down2 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([bottleneck_channels, 12, 12]),
            nn.ReLU()
        )  # 24→12
        self.down3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([bottleneck_channels, 6, 6]),
            nn.ReLU()
        )  # 12→6

    def forward(self, x):
        x = self.conv0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x  # shape: (B, bottleneck_channels, H/8, W/8) -> (B, bottleneck_channels, 6, 6)

# Separate convolutional decoder to expand back (6→48 spatial via 3 upsamples)
class ConvDecoder(nn.Module):
    def __init__(self, bottleneck_channels=16, out_channels=4):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([bottleneck_channels, 12, 12]),
            nn.ReLU()
        )  # 6→12
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([bottleneck_channels, 24, 24]),
            nn.ReLU()
        )  # 12→24
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([bottleneck_channels, 48, 48]),
            nn.ReLU()
        )  # 24→48
        self.conv_out = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(x)
        return x  # shape: (B, out_channels, H, W)

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoencoderKL(**config)
        load_checkpoint_cascast(
            "/home/vatsal/NWM/weather/pipeline/autoencoder_ckpt.pth", self.autoencoder
        )
        self.autoencoder.eval()
        for p in self.autoencoder.parameters(): p.requires_grad = False
        self.autoencoder.requires_grad_(False)
        self.scaling_factor = 0.18215
        self.autoencoder = torch.compile(self.autoencoder, mode='reduce-overhead')

    @torch.no_grad()
    def encode(self, x):
        B, T, C, H, W = x.shape
        out = []
        for i in range(T):
            z = self.autoencoder.encode(x[:, i]).mode()
            out.append(z.unsqueeze(1))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def decode(self, z):
        B, T, C, H, W = z.shape
        out = []
        for i in range(T):
            dec = self.autoencoder.decode(z[:, i])
            out.append(dec.unsqueeze(1))
        return torch.cat(out, dim=1)
    
class Model(pl.LightningModule):
    def __init__(self, autoencoder, input_frames=13, pred_frames=12, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['autoencoder'])
        self.autoencoder = autoencoder
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.lr = lr

        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

        self.to_bottleneck = nn.Linear(16 * 6 * 6, 512)  # 16 channels, 6x6 spatial
        self.from_bottleneck = nn.Linear(512, 16 * 6 * 6)
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(colored(f"Total trainable parameters: {total_params:,}", 'blue'))
        print(colored(f"Input frames: {input_frames}, Predicted frames: {pred_frames}", 'yellow'))
    
    def forward(self, x):
        B, LC, LH, LW = x.shape
        b = self.encoder(x)  # (B, bC, H/8, W/8)
        flat = b.reshape(B, -1)
        bottleneck = self.to_bottleneck(flat)  # (B, 512)

        from_bottleneck = self.from_bottleneck(bottleneck)  # (B, 16*6*6)
        from_bottleneck = from_bottleneck.reshape(B, 16, 6, 6)  # (B, 16, 6, 6)
        dec = self.decoder(from_bottleneck)
        return dec.reshape(B, LC, LH, LW)

    def training_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)
        v = v * self.autoencoder.scaling_factor  # (B, T, LC, LH, LW)
        inp = v[:, 0] # (B, LC, LH, LW)
        pred = self(inp)
        loss = F.mse_loss(pred, inp)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)
        v = v * self.autoencoder.scaling_factor  # (B, T, LC, LH, LW)
        inp = v[:, 0]
        pred = self(inp)
        loss = F.mse_loss(pred, inp)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        inp = v[:, 0].unsqueeze(1)  #[B, 1, LC, LH, LW]
        encoded = self.autoencoder.encode(inp).squeeze(1)  # (B, LC, LH, LW)
        encoded = encoded * self.autoencoder.scaling_factor
        pred = self(encoded) # (B, LC, LH, LW)

        pred /= self.autoencoder.scaling_factor 
        decoded_pred = self.autoencoder.decode(pred.unsqueeze(1)) # (B, 1, LC, LH, LW)

        metrics = calc_metrics(decoded_pred, inp)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)


    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=10, T_mult=2, eta_min=1e-6
                )        
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('medium')

    cfg = OmegaConf.load("/home/vatsal/NWM/weather/pipeline/configs/models/vae.yaml")
    vae = Autoencoder(cfg)
    
    from pytorch_lightning.callbacks import ModelCheckpoint

    name = "pae_ae"

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/{name}/checkpoints/",
        filename="{pae_ae}-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=1000,
    )

    wandb_logger = WandbLogger(project=name)

    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator='gpu', 
        devices=[0],
        gradient_clip_val=1.0,
        precision='bf16', 
        limit_test_batches=100,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    dm = SEVIRLightningDataModule(); dm.prepare_data(); dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader()]:
        for sample in loader:
            data = sample["vil"]
            print(f"Data shape: {data.shape}")
            break
    
    model = Model.load_from_checkpoint("/home/vatsal/NWM/weather/pipeline/logs/pae_ae/checkpoints/pae_ae=0-epoch=01-step=045000.ckpt", autoencoder=vae)
    trainer.test(model, dm)