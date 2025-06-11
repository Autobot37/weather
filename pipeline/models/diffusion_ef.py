from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from pipeline.modeldefinitions.ef import get_earthformer
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
"""
Earthformer takes B T H W C
"""
class DEarthformer(BaseModel):
    def __init__(self, data_config):
        super().__init__(model_name="Diffusion_Earthformer")
        config = OmegaConf.load("configs/models/earthformer.yaml")
        self.transformer, self.scheduler = get_earthformer(config["Model"]), DDIMScheduler(num_train_timesteps=config["timesteps"])
        self.mse_loss = nn.MSELoss()
        data_config = OmegaConf.load(data_config)
        self.in_window = data_config.in_window
        self.out_window = data_config.out_window
        self.num_train_timesteps = config["timesteps"]
        self.image_size = data_config.image_size; self.image_size = int(self.image_size)
        self.timestep_embedding = nn.Embedding(self.num_train_timesteps, self.image_size * self.image_size)
        
    def forward(self, noisy, timesteps, cond):
        H = self.image_size
        timesteps = self.timestep_embedding(timesteps) #[B, H * W]
        timesteps = timesteps.reshape(timesteps.size(0), 1, H, H, 1)
        inp = torch.concat([noisy, cond, timesteps], dim=1)
        return self.transformer(inp)
    
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
        vil = batch['vil'].permute(0,3,1,2).unsqueeze(4).float()
        vil0 = vil[:,0:1]
        vil = vil - vil0
        cond, tgt = vil[:,:self.in_window], vil[:,self.in_window:]
        noise = torch.randn_like(tgt)
        B = tgt.size(0)
        T = torch.randint(0, self.num_train_timesteps, (B,), device=self.device)
        noisy = self.scheduler.add_noise(tgt, noise, T)
        pred = self(noisy, T, cond)
        loss = self.mse_loss(pred, noise)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.global_step % 100 == 0:
            assert pred.dim() == 5 and pred.shape[-1] == 1, f"Expected pred shape (B, T, 1, H, W), got {pred.shape}"
            assert tgt.dim() == 5 and tgt.shape[-1] == 1, f"Expected tgt shape (B, T, 1, H, W), got {tgt.shape}"
            pred = pred.permute(0, 1, 4, 2, 3)  # (B, T, 1, H, W)
            tgt = tgt.permute(0, 1, 4, 2, 3)  # (B, T, 1, H, W)
            pred = pred.clamp(-1, 1).add(1).div(2).detach()  # Normalize to [0, 1]
            tgt = tgt.clamp(-1, 1).add(1).div(2).detach()  # Normalize to [0, 1]
            self.log_metrics(preds=pred, targets=tgt, stage="train")
            
            if self.global_step % 100 == 0:
                vil0 = vil0.permute(0, 1, 4, 2, 3)
                pred = (pred + vil0)/2
                tgt = (tgt + vil0)/2
                pred = pred.squeeze(2).cpu().numpy()
                tgt = tgt.squeeze(2).cpu().numpy()
                self.log_plots(preds = pred, targets=tgt, plot_fn = SEVIRLightningDataModule.plot_sample, label = f"Train_{self.current_epoch}_{self.global_step}")
        return loss

    def validation_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).unsqueeze(4).float()
        vil0 = vil[:,0:1]
        vil = vil - vil0
        cond, tgt = vil[:,:self.in_window], vil[:,self.in_window:]
        noise = torch.randn_like(tgt)
        B = tgt.size(0)
        T = torch.randint(0, self.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(tgt, noise, T)
        pred = self(noisy, T, cond)
        loss = self.mse_loss(pred, noise)
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        pred = pred.permute(0, 1, 4, 2, 3)  # (B, T, 1, H, W)
        tgt = tgt.permute(0, 1, 4, 2, 3)  # (B, T, 1, H, W)
        pred = pred.clamp(-1, 1).add(1).div(2).detach()  # Normalize to [0, 1]
        tgt = tgt.clamp(-1, 1).add(1).div(2).detach()  # Normalize to [0, 1]
        self.log_metrics(preds=pred, targets=tgt, stage="val")

        if self.global_step % 100 == 0:
            vil0 = vil0.permute(0, 1, 4, 2, 3)
            pred = (pred + vil0)/2
            tgt = (tgt + vil0)/2
            pred = pred.squeeze(2).cpu().numpy()
            tgt = tgt.squeeze(2).cpu().numpy()
            self.log_plots(preds = pred, targets = tgt, plot_fn = SEVIRLightningDataModule.plot_sample, label = f"Val_{self.current_epoch}_{self.global_step}")

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data();dm.setup()

    logger = WandbLogger(project="DEarthformer", save_dir="logs/DEarthformer")
    run_id = logger.version 
    print(f"Logger: {logger.name}, Run ID: {run_id}") 

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/DEarthformer/checkpoints/{run_id}",
        filename="DEarthformer-{epoch:02d}-{step:06d}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=1000,
        save_last=True,
        save_on_train_epoch_end=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=10,
        gpus = 1 if torch.cuda.is_available() else 0,
        logger=logger,
        val_check_interval=len(dm.train_dataloader()) // 2, 
        callbacks=[checkpoint_callback, lr_monitor], CodeLogger(),
    )
    from termcolor import colored
    for sample in dm.train_dataloader():
        data = sample['vil']
        print(colored(f"Sample shape: {data.shape}", 'blue'))
        print(colored(f"Min/Max: {data.min().item()}, {data.max().item()}", 'blue'))
        break

    model = DEarthformer("configs/datasets/sevir.yaml")
    print(colored("Model initialized!", 'green'))
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))