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

    def forward(self, x):
        # x: (B, T, C, H, W)
        enc = self.autoencoder.encode(x)
        enc_all = enc - enc[:, :1]
        cond = enc_all[:, :self.in_window]
        target = enc_all[:, self.in_window:]
        noise = torch.randn_like(target)
        B = noise.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        noisy = self.diffnet.scheduler.add_noise(target, noise, t)
        pred = self.diffnet(noisy, t, cond)
        loss = self.loss_fn(pred, noise)
        return pred, target, loss

    def training_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2).unsqueeze(2).float()
        preds, target, loss = self(x)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.global_step % 100 == 0:
            preds = self.autoencoder.decode(preds)
            target = self.autoencoder.decode(target)
            preds = preds.clamp(-1, 1).add(1).div(2)  # Normalize to [0, 1]
            target = target.clamp(-1, 1).add(1).div(2)  # Normalize to [0, 1]
            self.log_metrics(preds=preds, target=target, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2).unsqueeze(2).float()
        preds, target, loss = self(x)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        preds = self.autoencoder.decode(preds)
        target = self.autoencoder.decode(target)
        preds = preds.clamp(-1, 1).add(1).div(2)  # Normalize to [0, 1]
        target = target.clamp(-1, 1).add(1).div(2)  # Normalize to [0, 1]
        self.log_metrics(preds=preds, target=target, stage="val")
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

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
        devices=[0],
        logger=logger,
        val_check_interval=len(dm.train_dataloader()), 
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
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