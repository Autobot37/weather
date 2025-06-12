from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from pipeline.modeldefinitions.diffusion import *
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger

class Model(BaseModel):
    def __init__(self, data_config):
        super().__init__(model_name="purediff_res")
        self.mse_loss = nn.MSELoss()
        
        cfg = OmegaConf.load(data_config)
        self.in_window = cfg.in_window
        self.out_window = cfg.out_window
        self.image_size = int(cfg.image_size)
        self.model = create_diffusion_model(image_size=128, in_channels=self.in_window, timesteps=1000, sampling_timesteps=8, objective='pred_noise')
        
    def forward(self, target, cond):
        return self.model(target, cond=cond)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        total_steps = self.trainer.estimated_stepping_batches
        assert total_steps > 0, "Estimated stepping batches must be > 0"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }
    def lr_scheduler_step(self, scheduler, optimizer_idx, metrics):
        scheduler.step()

    def training_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).float()

        vil0 = vil.clone()[:, 0:1, :, :] 
        vil = vil - vil0
        vil = (vil + 1.0) / 2.0  # Normalize to [0, 1]
        cond, tgt = vil[:, :self.in_window], vil[:, self.in_window:]

        loss = self(tgt, cond)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if batch_idx % 100 == 0:
            with torch.no_grad():
                B, T, H, W = tgt.shape
                sampled = self.model.sample(shape=(B, T, H, W), cond=cond)
            sampled = sampled.detach(); tgt = tgt.detach()
            
            sampled = sampled * 2.0 - 1.0
            tgt = tgt * 2.0 - 1.0
            sampled = (sampled + vil0) 
            tgt = (tgt + vil0) 

            sampled_cpu = sampled.cpu()
            tgt_cpu = tgt.cpu()

            self.log_metrics(preds=sampled.unsqueeze(2), targets=tgt.unsqueeze(2), stage="train")

            self.log_plots(
                preds=sampled_cpu.numpy(), targets=tgt_cpu.numpy(),
                plot_fn=SEVIRLightningDataModule.plot_sample,
                label=f"Train_{self.current_epoch}_{self.global_step}"
            )
            # free GPU memory
            del sampled, sampled_cpu, tgt_cpu
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).float()
        vil0 = vil.clone()[:, 0:1, :, :] 
        vil = vil - vil0
        vil = (vil + 1.0) / 2.0  # Normalize to [0, 1]

        cond, tgt = vil[:, :self.in_window], vil[:, self.in_window:]

        loss = self(tgt, cond)
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        with torch.no_grad():
            B, T, H, W = tgt.shape
            sampled = self.model.sample(shape=(B, T, H, W), cond=cond)
        sampled = sampled.detach(); tgt = tgt.detach()
        
        sampled = sampled * 2.0 - 1.0
        tgt = tgt * 2.0 - 1.0
        sampled = (sampled + vil0) 
        tgt = (tgt + vil0) 
    
        sampled_cpu = sampled.cpu()
        tgt_cpu = tgt.cpu()

        self.log_metrics(preds=sampled.unsqueeze(2), targets=tgt.unsqueeze(2), stage="val")

        if batch_idx % 10 == 0:
            self.log_plots(
                preds=sampled_cpu.numpy(), targets=tgt_cpu.numpy(),
                plot_fn=SEVIRLightningDataModule.plot_sample,
                label=f"Val_{self.current_epoch}_{self.global_step}"
            )
            del sampled, sampled_cpu, tgt_cpu
            torch.cuda.empty_cache()

        return loss

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data()
    dm.setup()

    logger = WandbLogger(project="purediff_res", save_dir="logs/purediff_res")
    run_id = logger.version
    print(f"Logger: {logger.name}, Run ID: {run_id}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"logs/purediff_res/checkpoints/{run_id}",
        filename="purediff_res-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=[1],  # ensure GPU with free memory
        logger=logger,
        val_check_interval=len(dm.train_dataloader()),
        callbacks=[checkpoint_callback, lr_monitor]
    )

    from termcolor import colored
    for sample in dm.train_dataloader():
        print(colored(f"Sample shape: {sample['vil'].shape}", 'blue'))
        print(colored(f"Min/Max: {sample['vil'].min().item()}, {sample['vil'].max().item()}", 'blue'))
        break

    model = Model("configs/datasets/sevir.yaml")
    print(colored("Model initialized!", 'green'))
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))
