from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from CasCast.networks.prediff.taming.autoencoder_kl import AutoencoderKL
from pipeline.modeldefinitions.dit import CasFormer
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pipeline.utils import load_checkpoint_cascast
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger

class DiffusionUtils:
    def __init__(self, scheduler, num_inference_steps=50):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        
    def add_noise(self, clean_images, noise, timesteps):
        return self.scheduler.add_noise(clean_images, noise, timesteps)
    
    @torch.no_grad()
    def sample(self, model, shape, cond, device):
        # shape: (B, T, C, H, W)
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        
        # Start with random noise
        sample = torch.randn(shape, device=device)
        
        for timestep in self.scheduler.timesteps:
            timestep_batch = timestep.expand(shape[0]).to(device)
            noise_pred = model(sample, timestep_batch, cond)
            sample = self.scheduler.step(noise_pred, timestep, sample).prev_sample
            
        return sample

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CasFormer(arch='DiT-custom', config=config.Model)
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon')

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        return self.model(noisy, timesteps, cond)

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoencoderKL(**config)
        self.autoencoder.eval() 
        self.scaling, self.shift = (0.0259326533906051, 0.30854346410703587)
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
            frame = x[:, i]  # (B, C, H, W)
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

class Diffusion(BaseModel):
    def __init__(self):
        super().__init__(model_name="DiT")
        dit_config = OmegaConf.load("configs/models/dit.yaml")
        vae_config = OmegaConf.load("configs/models/vae.yaml")
        self.diffnet = DiT(dit_config)
        self.diffusion_utils = DiffusionUtils(self.diffnet.scheduler, num_inference_steps=50)
        self.autoencoder = Autoencoder(vae_config)
        data_config = OmegaConf.load("configs/datasets/sevir2.yaml")
        self.in_window = data_config.in_window
        self.num_timesteps = dit_config.timesteps
        self.loss_fn = nn.MSELoss()

    def forward(self, noisy, timesteps, cond):
        return self.diffnet(noisy, timesteps, cond)

    def training_step(self, batch, batch_idx):
        # Process input: (B, H, W, T) -> (B, T, C, H, W)
        x = batch['vil'].permute(0,3,1,2).unsqueeze(2).float()
        enc = self.autoencoder.encode(x)
        enc = enc * self.autoencoder.scaling + self.autoencoder.shift  

        cond = enc[:, :self.in_window]
        targets = enc[:, self.in_window:]
        noise = torch.randn_like(targets)
        B = noise.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        
        noisy = self.diffusion_utils.add_noise(targets, noise, t)
        pred = self(noisy, t, cond)
        loss = self.loss_fn(pred, noise)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % 500 == 0:
            sampled = self.diffusion_utils.sample(self, targets.shape, cond, self.device)
            sampled = sampled / self.autoencoder.scaling - self.autoencoder.shift
            targets = targets / self.autoencoder.scaling - self.autoencoder.shift

            sampled_decoded = self.autoencoder.decode(sampled).detach().clamp(0, 1)
            targets_decoded = self.autoencoder.decode(targets).detach().clamp(0, 1)
            self.log_metrics(preds=sampled_decoded, targets=targets_decoded, stage="train")

            # Plot: (B, T, C, H, W) -> (B, T, H, W)
            sampled_plot = sampled_decoded.squeeze(2).cpu().numpy()
            targets_plot = targets_decoded.squeeze(2).cpu().numpy()
            self.log_plots(preds=sampled_plot, targets=targets_plot, 
                            plot_fn=SEVIRLightningDataModule.plot_sample, 
                            label=f"train_{self.current_epoch}_{self.global_step}")

        return loss

    def validation_step(self, batch, batch_idx):
        # Process input: (B, H, W, T) -> (B, T, C, H, W)
        x = batch['vil'].permute(0,3,1,2).unsqueeze(2).float()
        enc = self.autoencoder.encode(x)
        enc *= self.autoencoder.scaling_factor  
        enc = enc.clamp(0, 1)

        cond = enc[:, :self.in_window]
        targets = enc[:, self.in_window:]
        
        noise = torch.randn_like(targets)
        B = noise.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        noisy = self.diffusion_utils.add_noise(targets, noise, t)
        pred = self(noisy, t, cond)
        loss = self.loss_fn(pred, noise)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        sampled = self.diffusion_utils.sample(self, targets.shape, cond, self.device)
        
        sampled /= self.autoencoder.scaling_factor
        targets /= self.autoencoder.scaling_factor

        sampled_decoded = self.autoencoder.decode(sampled).detach().clamp(0, 1)
        targets_decoded = self.autoencoder.decode(targets).detach().clamp(0, 1)
        
        self.log_metrics(preds=sampled_decoded, targets=targets_decoded, stage="val")
        
        # Plot: (B, T, C, H, W) -> (B, T, H, W)
        if batch_idx % 50 == 0:
            sampled_plot = sampled_decoded.squeeze(2).cpu().numpy()
            targets_plot = targets_decoded.squeeze(2).cpu().numpy()
            self.log_plots(preds=sampled_plot, targets=targets_plot, 
                            plot_fn=SEVIRLightningDataModule.plot_sample, 
                            label=f"val_{self.current_epoch}_{self.global_step}")
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
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
            }
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data()
    dm.setup()

    logger = WandbLogger(project="DiT", save_dir="logs/DiT", name = "scaleshift")
    run_id = "scaleshift" 
    print(f"Logger: {logger.name}, Run ID: {run_id}") 

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/DiT/checkpoints/{run_id}",
        filename="DiT-{epoch:02d}-{step:06d}",
        save_last=True,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=[0],
        logger=logger,
        val_check_interval=len(dm.train_dataloader()), 
        callbacks=[checkpoint_callback, lr_monitor, CodeLogger()],
        limit_val_batches=100,
    )
    
    from termcolor import colored
    for sample in dm.train_dataloader():
        data = sample['vil']
        print(colored(f"Sample shape: {data.shape}", 'blue'))
        print(colored(f"Min/Max: {data.min().item()}, {data.max().item()}", 'blue'))
        break

    model = Diffusion()
    print(colored("Model initialized!", 'green'))
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))