import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule
from pipeline.utils import calc_metrics

os.environ['WANDB_API_KEY'] = '041eda3850f131617ee1d1c9714e6230c6ac4772'

class PersistenceModel(pl.LightningModule):
    def __init__(self, input_frames: int):
        super().__init__()
        self.input_frames = input_frames

    def test_step(self, batch, batch_idx):
        # batch['vil']: (B, H, W, T)
        v = batch['vil'].permute(0, 3, 1, 2).unsqueeze(2).float()  # -> (B, T, C=1, H, W)
        inp = v[:, :self.input_frames]                           
        tgt = v[:, self.input_frames:]                             

        last_frame = inp[:, -1:]                                   # (B, 1, C, H, W)
        pred = last_frame.expand(-1, tgt.shape[1], -1, -1, -1)     # (B, T_future, C, H, W)

        metrics = calc_metrics(pred, tgt)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

if __name__ == '__main__':
    INPUT_FRAMES = 13

    dm = SEVIRLightningDataModule()
    dm.prepare_data()
    dm.setup()

    wandb_logger = WandbLogger(project='persistence-baseline')

    model = PersistenceModel(input_frames=INPUT_FRAMES)

    trainer = pl.Trainer(
        max_epochs=1,
        logger=wandb_logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=1,
    )

    trainer.test(model, datamodule=dm)
