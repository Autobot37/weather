from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from pipeline.modeldefinitions.ef import get_earthformer
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from pipeline.modeldefinitions.unet3d import *
from CasCast.networks.prediff.taming.autoencoder_kl import AutoencoderKL
from pipeline.utils import load_checkpoint_cascast

class DiffusionUtils:
    def __init__(self, scheduler, num_inference_steps=10):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        
    def add_noise(self, clean_images, noise, timesteps):
        return self.scheduler.add_noise(clean_images, noise, timesteps)
    
    @torch.no_grad()
    def sample(self, model, shape, cond, device):
        # shape: (B, T, H, W, C)
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        
        # Start with random noise
        sample = torch.randn(shape, device=device)
        
        for timestep in self.scheduler.timesteps:
            timestep_batch = timestep.expand(shape[0]).to(device)
            
            # Predict noise
            noise_pred = model(sample, timestep_batch, cond)
            # Denoise step
            sample = self.scheduler.step(noise_pred, timestep, sample).prev_sample
            
        return sample


class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoencoderKL(**config)
        self.autoencoder.eval() 
        self.scaling_factor = 0.18215
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

class Model(BaseModel):
    def __init__(self, data_config):
        super().__init__(model_name="HFUnet3d")
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon')
        self.diffusion_utils = DiffusionUtils(self.scheduler, num_inference_steps=50)
        self.loss_fn = nn.MSELoss()
        
        data_config = OmegaConf.load(data_config)
        self.in_window = data_config.in_window
        self.out_window = data_config.out_window
        self.num_train_timesteps = 1000
        self.image_size = data_config.image_size
        self.image_size = int(self.image_size)
        in_channels = self.in_window
        self.unet = CustomUNet3D()
        vae_config = OmegaConf.load("configs/models/vae.yaml")
        self.autoencoder = Autoencoder(vae_config)
        self.num_timesteps = 1000

    def forward(self, noisy, timesteps, cond):
        # noisy: (B, Tn, C, H, W), cond : (B, Tc, C, H, W)
        out = self.unet(noisy, timesteps, cond)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-2)
        total_steps = self.trainer.estimated_stepping_batches
        assert total_steps > 0, "Estimated stepping batches must be greater than 0."
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
        # Process input: (B, H, W, T) -> (B, T, H, W)
        vil = batch['vil'].permute(0,3,1,2).float()
        vil = vil.unsqueeze(2)  # (B, T, 1, H, W)
        enc = self.autoencoder.encode(vil)  # (B, T, latent_C, H, W)
        enc *= self.autoencoder.scaling_factor 
        cond, targets = enc[:,:self.in_window], enc[:,self.in_window:]
        noise = torch.randn_like(targets)
        B = targets.size(0)
        T = torch.randint(0, self.num_train_timesteps, (B,), device=self.device)
        
        noisy = self.diffusion_utils.add_noise(targets, noise, T)
        pred = self(noisy, T, cond)
        loss = self.loss_fn(pred, noise)
        
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                sampled = self.diffusion_utils.sample(self, targets.shape, cond, self.device)
                sampled = sampled / self.autoencoder.scaling_factor
                targets = targets / self.autoencoder.scaling_factor

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

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data()
    dm.setup()

    logger = WandbLogger(project="HFUnet3d", save_dir="logs/HFUnet3d", name = 'test')
    run_id = logger.version 
    print(f"Logger: {logger.name}, Run ID: {run_id}") 

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/HFUnet3d/checkpoints/{run_id}",
        filename="HFUnet3d-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=[1],
        logger=logger,
        val_check_interval=len(dm.train_dataloader()) // 2, 
        callbacks=[checkpoint_callback, lr_monitor, CodeLogger()],
        limit_val_batches=100
    )
    
    from termcolor import colored
    for sample in dm.train_dataloader():
        data = sample['vil']
        print(colored(f"Sample shape: {data.shape}", 'blue'))
        print(colored(f"Min/Max: {data.min().item()}, {data.max().item()}", 'blue'))
        break

    model = Model("configs/datasets/sevir2.yaml")
    print(colored("Model initialized!", 'green'))
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))