from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from neuralop.models import FNO
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

class DFNO(BaseModel):
    def __init__(self):
        super().__init__(model_name="DiT_VAE")
        data_config = OmegaConf.load("configs/datasets/sevir.yaml")
        self.in_window = data_config.in_window
        self.out_window = data_config.out_window
        self.in_ch = self.in_window + self.out_window
        self.num_timesteps = 250
    
        self.fno = FNO(n_modes=(32, 32), in_channels=self.in_ch, out_channels=self.out_window, hidden_channels=32)
        self.scheduler = DDIMScheduler(num_train_timesteps=self.num_timesteps)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (B, T, H, W)
        x0 = x[:, 0:1, :, :].clone()  
        x = x - x0
        cond = x[:, :self.in_window]
        target = x[:, self.in_window:]
        noise = torch.randn_like(target)
        B = noise.size(0)
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        noisy = self.scheduler.add_noise(target, noise, t)
        inp = torch.cat([noisy, cond], dim=1)  # (B, IN + OUT, H, W)
        pred = self.fno(inp)
        loss = self.loss_fn(pred, noise)
        return pred, target, loss

    def training_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2).float()
        preds, target, loss = self(x)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.global_step % 100 == 0:
            preds = preds.unsqueeze(2)  # (B, OUT, 1, H, W)
            target = target.unsqueeze(2)  # (B, OUT, 1, H, W)
            self.log_metrics(preds, target, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2).float()
        preds, target, loss = self(x)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        preds = preds.unsqueeze(2)  # (B, OUT, 1, H, W)
        target = target.unsqueeze(2)  # (B, OUT, 1, H, W)
        self.log_metrics(preds, target, stage="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        return super().lr_scheduler_step(scheduler, optimizer_idx, metric)

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data();dm.setup()

    logger = WandbLogger(project="DFNO", save_dir="logs/DFNO")
    run_id = logger.version 
    print(f"Logger: {logger.name}, Run ID: {run_id}") 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/DFNO/checkpoints/{run_id}",
        filename="DFNO-{epoch:02d}-{step:06d}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=1001,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=2,
        gpus = 2 if torch.cuda.is_available() else 0,
        logger=logger,
        strategy= "ddp",
        limit_train_batches=2000,
        limit_val_batches=100,
        val_check_interval=1000, 
        callbacks=[checkpoint_callback, lr_monitor, CodeLogger()],
    )
    from termcolor import colored
    for sample in dm.train_dataloader():
        data = sample['vil']
        print(colored(f"Sample shape: {data.shape}", 'blue'))
        print(colored(f"Min/Max: {data.min().item()}, {data.max().item()}", 'blue'))
        break

    model = DFNO()
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))