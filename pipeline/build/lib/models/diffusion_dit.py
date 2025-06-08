from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from pipeline.modeldefinitions.dit import CasFormer
from basemodel import *

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = CasFormer(arch='DiT-custom', config=config.Model)
        self.scheduler = DDIMScheduler(num_train_timesteps=config.timesteps)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        return self.model(noisy, timesteps, cond)

class Diffusion(BaseModel):
    def __init__(self):
        super().__init__(model_name="DiT_VAE")
        dit_config = OmegaConf.load("configs/models/dit_vae.yaml")
        self.diffnet = DiT(dit_config)
        data_config = OmegaConf.load("configs/datasets/sevir.yaml")
        self.in_window = data_config.in_window
        self.num_timesteps = dit_config.timesteps
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (B, T, C, H, W)
        x0 = x[:, 0:1, :, :, :]  
        x -= x0
        cond = x[:, :self.in_window]
        target = x[:, self.in_window:]
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
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_metrics(preds, target, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2).unsqueeze(2).float()
        preds, target, loss = self(x)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_metrics(preds, target, stage="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data();dm.setup()

    logger = BaseModel.get_logger(model_name="DiT-VAE", run_name="run1", save_dir="logs")
    trainer = Trainer(
        max_epochs=1,
        gpus=2 if torch.cuda.is_available() else 0,
        logger=logger,
        log_every_n_steps=100,
        strategy= "ddp"
    )
    from termcolor import colored
    for sample in dm.train_dataloader():
        data = sample['vil']
        print(colored(f"Sample shape: {data.shape}", 'blue'))
        print(colored(f"Min/Max: {data.min().item()}, {data.max().item()}", 'blue'))
        break

    model = Diffusion()
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))