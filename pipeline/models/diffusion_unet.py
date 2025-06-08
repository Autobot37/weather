from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from diffusers import DDIMScheduler
import torch.optim as optim
from modeldefinitions.unet2d import get_unet2d
from basemodel import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

class DUNet2D(BaseModel):
    def __init__(self, data_config):
        super().__init__(model_name="DUNet2D")
        data_config = OmegaConf.load(data_config)
        self.in_window = data_config.in_window
        self.out_window = data_config.out_window
        self.image_size = data_config.image_size
        config = OmegaConf.load("configs/models/unet.yaml")
        self.num_train_timesteps = config.timesteps

        in_ch = self.in_window + self.out_window
        self.unet = get_unet2d(image_size=self.image_size, in_ch=in_ch, out_window=self.out_window, num_train_timesteps=self.num_train_timesteps)
        self.scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps)

        self.loss_fn = nn.MSELoss()

    def forward(self, noisy, timesteps, cond):
        x = torch.cat((noisy, cond), dim=1)
        return self.unet(x, timesteps).sample

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).float()
        vil0 = vil[:,0:1]
        vil = vil - vil0
        cond, tgt = vil[:,:self.in_window], vil[:,self.in_window:]
        noise = torch.randn_like(tgt)
        B = tgt.size(0)
        T = torch.randint(0, self.num_train_timesteps, (B,), device=self.device)
        noisy = self.scheduler.add_noise(tgt, noise, T)
        pred = self(noisy, T, cond)
        loss = self.mse_loss(pred, noise)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).float()
        vil0 = vil[:,0:1]
        vil = vil - vil0
        cond, tgt = vil[:,:self.in_window], vil[:,self.in_window:]
        B = tgt.size(0)

        gen = torch.randn(B, self.out_window, vil.size(2), vil.size(3), device=self.device)
        self.scheduler.set_timesteps(self.num_train_timesteps)
        for t in self.scheduler.timesteps:
            tb = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps = self(gen, tb, cond)
            gen = self.scheduler.step(eps, t, gen).prev_sample

        gen = gen.clamp(-1,1).add(1).div(2)
        tgt = tgt.clamp(-1,1).add(1).div(2)
        self.log_metrics(preds = gen, target = tgt, stage = "val")

if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data();dm.setup()

    logger = BaseModel.get_logger(model_name="DFNO", save_dir="logs")
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
        callbacks=[checkpoint_callback, lr_monitor],
    )
    from termcolor import colored
    for sample in dm.train_dataloader():
        data = sample['vil']
        print(colored(f"Sample shape: {data.shape}", 'blue'))
        print(colored(f"Min/Max: {data.min().item()}, {data.max().item()}", 'blue'))
        break

    model = DUNet2D()
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))