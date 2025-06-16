import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored

from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from pipeline.models.ae import ConvAutoencoder
from pipeline.utils import calc_metrics

"""
384x384
"""

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = ConvAutoencoder()
        self.autoencoder.load_state_dict(torch.load("/home/vatsal/NWM/weather/pipeline/tb_logs/autoencoder/version_11/checkpoints/epoch=19-step=150540.ckpt")["state_dict"], strict=False)
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        print(colored("Autoencoder loaded and set to eval mode", 'green'))

    @torch.no_grad()
    def encode(self, x):
        B, T, C, H, W = x.shape
        out = []
        for i in range(T):
            z = self.autoencoder(x[:, i])[1]
            out.append(z.unsqueeze(1))
        return torch.cat(out, dim=1)
    
    @torch.no_grad()
    def decode(self, z):
        B, T, C, H, W = z.shape
        out = []
        for i in range(T):
            x = self.autoencoder.decode(z[:, i])
            out.append(x.unsqueeze(1))
        return torch.cat(out, dim=1)
    
class SimpleMLP(nn.Module):
    def __init__(self, input_dim = 256, output_dim = 256, hidden_dim=512, num_layers=3):
        super().__init__()
        layers = []
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        
        for _ in range(num_layers-2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        return self.net(x)

class MLPFramePredictor(pl.LightningModule):
    def __init__(self, autoencoder, input_frames=10, pred_frames=10,
                 hidden_dim=512, num_layers=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['autoencoder'])
        self.autoencoder = autoencoder
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.lr = lr

        self.predictor = SimpleMLP()
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(colored(f"Total trainable parameters: {total_params:,}", 'blue'))
        print(colored(f"Input frames: {input_frames}, Predicted frames: {pred_frames}", 'yellow'))

    def forward(self, x):
        out = self.predictor(x)
        return out

    def training_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)  # (B, T, LC, LH, LW)

        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t

        pred = self(inp)
        loss = F.mse_loss(pred, tgt)
        self.log('train_loss', loss, prog_bar=True)
        self.log('min_inp', inp.min(), prog_bar=True)
        self.log('max_inp', inp.max(), prog_bar=True)
        self.log('min_pred', pred.min(), prog_bar=True)
        self.log('max_pred', pred.max(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)

        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t

        pred = self(inp)
        loss = F.mse_loss(pred, tgt)
        self.log('val_loss', loss, prog_bar=True)
        self.log('min_inp', inp.min(), prog_bar=True)
        self.log('max_inp', inp.max(), prog_bar=True)
        self.log('min_pred', pred.min(), prog_bar=True)
        self.log('max_pred', pred.max(), prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        v = batch['vil'].permute(0,3,1,2).unsqueeze(2)
        v = self.autoencoder.encode(v)

        inp, tgt = v[:, :self.input_frames], v[:, self.input_frames:]
        inp_t = inp[:, -1].unsqueeze(1)
        inp = inp - inp_t
        tgt = tgt - inp_t

        pred = self(inp)
        
        pred = pred + inp_t
        tgt = tgt + inp_t

        decoded_pred = self.autoencoder.autoencoder.decode(pred)
        decoded_tgt = self.autoencoder.autoencoder.decode(tgt)

        metrics = calc_metrics(decoded_pred, decoded_tgt)
        for k, v in metrics.items():
            self.log(f'test_{k}', v, prog_bar=True, on_step=True, on_epoch=True)

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/pae_me8/checkpoints/",
        filename="pae_me8-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=1000,
    )

    trainer = pl.Trainer(
        max_epochs=50, 
        accelerator='gpu', 
        devices=[1],
        gradient_clip_val=1.0,
        precision='bf16', 
        limit_test_batches=200,
        callbacks=[checkpoint_callback],
    )

    dm = SEVIRLightningDataModule(); dm.prepare_data(); dm.setup()
    for loader in [dm.train_dataloader(), dm.val_dataloader()]:
        for sample in loader:
            data = sample["vil"]
            print(f"Data shape: {data.shape}")
            break
    
    model = MLPFramePredictor.load_from_checkpoint("/home/vatsal/NWM/weather/pipeline/logs/pae_me8/checkpoints/pae_me8-epoch=01-step=008000.ckpt", autoencoder = vae)
    trainer.test(model, dm)