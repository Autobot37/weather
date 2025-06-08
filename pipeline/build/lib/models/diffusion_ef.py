from diffusers import DDIMScheduler
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from modeldefinitions.earthformer import get_earthformer
from basemodel import *

class DEarthformer(BaseModel):
    def __init__(self, data_config):
        super().__init__(model_name="Diffusion_Earthformer")
        config = OmegaConf.load("configs/models/earthformer.yaml")
        self.transformer, self.scheduler = get_earthformer(config["Model"]), DDIMScheduler(num_train_timesteps=config["timesteps"])
        self.loss_fn = nn.MSELoss()
        data_config = OmegaConf.load(data_config)
        self.in_window = data_config.in_window
        self.out_window = data_config.out_window
        self.num_train_timesteps = config["timesteps"]

    def forward(self, noisy, timesteps, cond):
        inp = torch.concat([noisy, cond], dim=1)
        return self.transformer(inp)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

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
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vil = batch['vil'].permute(0,3,1,2).unsqueeze(4).float()
        vil0 = vil[:,0:1]; vil = vil - vil0
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
        self.log_metrics(preds=gen, target=tgt, stage="val")


if __name__ == "__main__":
    seed_everything(42)
    dm = SEVIRLightningDataModule()
    dm.prepare_data();dm.setup()

    logger = BaseModel.get_logger(model_name="DEarthformer", run_name="run1", save_dir="logs")
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

    model = DEarthformer()
    trainer.fit(model, dm)
    print(colored("Training complete!", 'green'))