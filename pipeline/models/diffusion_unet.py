from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from diffusers import DDIMScheduler
import torch.optim as optim
from weather.pipeline.modeldefinitions.unet3d import CustomUNet3D
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pipeline.utils import load_checkpoint_cascast
from CasCast.networks.prediff.taming.autoencoder_kl import AutoencoderKL

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoencoderKL(**config)
        self.autoencoder.eval() 
        load_checkpoint_cascast("autoencoder_ckpt.pth", self.autoencoder)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.requires_grad_(False)
    
    @torch.no_grad()
    def encode(self, x):
        # x: (B, T, C, H, W)
        B, T, _, H, W = x.shape
        out = []
        for i in range(T):
            frame = x[:, i]  # [B, C, H, W]
            z = self.autoencoder.encode(frame).sample()
            out.append(z.unsqueeze(1))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def decode(self, x):
        # x: (B, T, latent_C, H, W)
        B, T, C, H, W = x.shape
        out = []
        for i in range(T):
            frame = x[:, i]
            dec = self.autoencoder.decode(frame)
            out.append(dec.unsqueeze(1))
        return torch.cat(out, dim=1)
    
class DUNet2D(BaseModel):
    def __init__(self, data_config):
        super().__init__(model_name="DUNet3D")
        data_config = OmegaConf.load(data_config)
        self.in_window = data_config.in_window
        self.out_window = data_config.out_window
        self.image_size = data_config.image_size
        config = OmegaConf.load("configs/models/unet.yaml")
        self.num_train_timesteps = config.timesteps

        self.unet = CustomUNet3D()
        vae_config = OmegaConf.load("configs/models/vae.yaml")
        self.autoencoder = Autoencoder(vae_config)        
        self.scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps)
        self.loss_fn = nn.MSELoss()

    def forward(self, noisy, timesteps, cond):
        sample =  self.unet(noisy, timesteps, cond).sample # (B, out_channels, T, H, W)
        return sample.permute(0, 2, 1, 3, 4)  # (B, T, out_channels, H, W)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-2)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def training_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).unsqueeze(2).float() # (B, T, C, H, W)
        vil = self.autoencoder.encode(vil)  # (B, T, C, H, W)
        vil0 = vil[:,0:1] # (B, 1, C, H, W)
        vil = vil - vil0
        cond, tgt = vil[:,:self.in_window], vil[:,self.in_window:]
        noise = torch.randn_like(tgt)
        B = tgt.size(0)
        T = torch.randint(0, self.num_train_timesteps, (B,), device=self.device)
        noisy = self.scheduler.add_noise(tgt, noise, T)
        pred = self(noisy, T, cond) # [B, T, C, H, W]
        loss = self.loss_fn(pred, noise)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.global_step % 100 == 0:
            pred = pred.clamp(-1,1).add(1).div(2)
            tgt = tgt.clamp(-1,1).add(1).div(2)
            pred = self.autoencoder.decode(pred)
            tgt = self.autoencoder.decode(tgt)
            self.log_metrics(preds=pred, target=tgt, stage="train")

            if self.global_step % 2000 == 0:
                pred = (pred + vil0)/2
                tgt = (tgt + vil0)/2
                pred = pred.squeeze(2).cpu().numpy()
                tgt = tgt.squeeze(2).cpu().numpy()
                self.log_plots(preds = pred, targets=tgt, plot_fn = SEVIRLightningDataModule.plot_sample, label = f"Train_{self.current_epoch}_{self.global_step}")

        return loss

    def validation_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).unsqueeze(2).float() # (B, T, C, H, W)
        vil = self.autoencoder.encode(vil)  # (B, T, C, H, W)
        vil0 = vil[:,0:1]
        vil = vil - vil0
        cond, tgt = vil[:,:self.in_window], vil[:,self.in_window:]
        noise = torch.randn_like(tgt)
        B = tgt.size(0)
        T = torch.randint(0, self.num_train_timesteps, (B,), device=self.device)
        noisy = self.scheduler.add_noise(tgt, noise, T)
        pred = self(noisy, T, cond)
        loss = self.loss_fn(pred, noise)
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        pred = pred.clamp(-1,1).add(1).div(2)
        tgt = tgt.clamp(-1,1).add(1).div(2)
        pred = self.autoencoder.decode(pred)
        tgt = self.autoencoder.decode(tgt)
        self.log_metrics(preds=pred, target=tgt, stage="val")

        if self.global_step % 2000 == 0:
            vil0 = vil0.permute(0, 1, 4, 2, 3)
            pred = (pred + vil0)/2
            tgt = (tgt + vil0)/2
            pred = pred.squeeze(2).cpu().numpy()
            tgt = tgt.squeeze(2).cpu().numpy()
            self.log_plots(preds = pred, targets = tgt, plot_fn = SEVIRLightningDataModule.plot_sample, label = f"Val_{self.current_epoch}_{self.global_step}")
            
        return loss

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data();dm.setup()
    dm.setup()

    logger = WandbLogger(project="DUnet", save_dir="logs/DUNet")
    run_id = logger.version 
    print(f"Logger: {logger.name}, Run ID: {run_id}") 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/DUNet2d/checkpoints/{run_id}",
        filename="DUNet2d-{epoch:02d}-{step:06d}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=1000,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=[1],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, CodeLogger()],
        precision=16
    )
    from termcolor import colored
    for sample in dm.train_dataloader():
        data = sample['vil']
        print(colored(f"Sample shape: {data.shape}", 'blue'))
        print(colored(f"Min/Max: {data.min().item()}, {data.max().item()}", 'blue'))
        break

    model = DUNet2D("configs/datasets/sevir.yaml")
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))