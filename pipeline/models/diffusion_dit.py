from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from CasCast.networks.prediff.taming.autoencoder_kl import AutoencoderKL
from pipeline.modeldefinitions.dit import CasFormer
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pipeline.utils import load_checkpoint_cascast

"""
B T C H W   
"""
class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CasFormer(arch='DiT-custom', config=config.Model)
        self.scheduler = DDIMScheduler(num_train_timesteps=config.timesteps)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        return self.model(noisy, timesteps, cond)

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

class Diffusion(BaseModel):
    def __init__(self):
        super().__init__(model_name="DiT")
        dit_config = OmegaConf.load("configs/models/dit.yaml")
        vae_config = OmegaConf.load("configs/models/vae.yaml")
        self.diffnet = DiT(dit_config)
        self.autoencoder = Autoencoder(vae_config)
        data_config = OmegaConf.load("configs/datasets/sevir.yaml")
        self.in_window = data_config.in_window
        self.num_timesteps = dit_config.timesteps
        self.loss_fn = nn.MSELoss()

    def forward(self, enc):
        # x: (B, T, C, H, W)
        cond = enc[:, :self.in_window]
        targets = enc[:, self.in_window:]
        noise = torch.randn_like(targets)
        B = noise.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        noisy = self.diffnet.scheduler.add_noise(targets, noise, t)
        pred = self.diffnet(noisy, t, cond)
        loss = self.loss_fn(pred, noise)
        return pred, targets, loss

    def training_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2).unsqueeze(2).float()
        enc = self.autoencoder.encode(x)
        vil0 = enc[:, 0:1]  
        enc = enc - vil0  
        preds, targets, loss = self(enc)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.global_step % 100 == 0:
            preds = self.autoencoder.decode(preds)
            targets = self.autoencoder.decode(targets)
            preds = preds.clamp(-1, 1).add(1).div(2).detach()  # Normalize to [0, 1]
            targets = targets.clamp(-1, 1).add(1).div(2).detach()  # Normalize to [0, 1]
            self.log_metrics(preds=preds, targets=targets, stage="train")

            if self.global_step % 2000 == 0:
                vil0 = self.autoencoder.decode(vil0)
                preds = (preds + vil0)/2
                targets = (targets + vil0)/2
                preds = preds.squeeze(2).cpu().numpy()
                targets = targets.squeeze(2).cpu().numpy()
                self.log_plots(preds = preds, targets=targets, plot_fn = SEVIRLightningDataModule.plot_sample, label = f"Train_{self.current_epoch}_{self.global_step}")
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2).unsqueeze(2).float()
        enc = self.autoencoder.encode(x)
        vil0 = enc[:, 0:1]
        enc = enc - vil0
        preds, targets, loss = self(enc)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        preds = self.autoencoder.decode(preds)
        targets = self.autoencoder.decode(targets)
        preds = preds.clamp(-1, 1).add(1).div(2)  # Normalize to [0, 1]
        targets = targets.clamp(-1, 1).add(1).div(2)  # Normalize to [0, 1]
        self.log_metrics(preds=preds, targets=targets, stage="val")

        if self.global_step % 1000 == 0:
            vil0 = self.autoencoder.decode(vil0)
            preds = (preds + vil0)/2
            targets = (targets + vil0)/2
            preds = preds.squeeze(2).cpu().numpy()
            targets = targets.squeeze(2).cpu().numpy()
            self.log_plots(preds = preds, targets=targets, plot_fn = SEVIRLightningDataModule.plot_sample, label = f"val_{self.current_epoch}_{self.global_step}")

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
                'frequency': 10,
            }
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data();dm.setup()

    logger = BaseModel.get_logger(model_name="DiT", save_dir="logs")
    run_id = logger.version 
    print(f"Logger: {logger.name}, Run ID: {run_id}") 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/DiT/checkpoints/{run_id}",
        filename="DiT-{epoch:02d}-{step:06d}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=1000,
        save_last=True,
        verbose = True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=[1],
        logger=logger,
        val_check_interval=len(dm.train_dataloader()), 
        callbacks=[checkpoint_callback, lr_monitor],
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