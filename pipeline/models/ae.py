import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT

import matplotlib.pyplot as plt

from pipeline.datasets.dataset_sevir import SEVIRLightningDataModule


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_dim=256):
        super().__init__()
        # Encoder with Residual Blocks
        layers = []
        channels = in_channels
        for i in range(6):
            out_ch = base_channels * (2 ** i)
            layers += [
                nn.Conv2d(channels, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(out_ch)
            ]
            channels = out_ch
        self.encoder = nn.Sequential(*layers)

        # compute shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 384, 384)
            h = self.encoder(dummy)
        c, h_, w_ = h.shape[1:]
        self.enc_shape = (c, h_, w_)
        features = c * h_ * w_

        # bottleneck
        self.fc1 = nn.Linear(features, latent_dim)
        self.fc2 = nn.Linear(latent_dim, features)

        # Decoder with skip-style ResidualUpsample
        rev_layers = []
        for i in reversed(range(5)):
            in_ch = base_channels * (2 ** i)
            rev_layers += [
                nn.ConvTranspose2d(channels, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(in_ch)
            ]
            channels = in_ch
        rev_layers += [nn.ConvTranspose2d(base_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                       nn.Sigmoid()]
        
        self.decoder = nn.Sequential(*rev_layers)

    def forward(self, x):
        enc = self.encoder(x)
        batch = enc.size(0)
        flat = enc.flatten(1)
        z = self.fc1(flat)
        flat_rec = self.fc2(z)
        dec_in = flat_rec.view(batch, *self.enc_shape)
        recon = self.decoder(dec_in)
        return recon, z
    
    def decode(self, z):
        batch = z.size(0)
        flat_rec = self.fc2(z)
        dec_in = flat_rec.view(batch, *self.enc_shape)
        recon = self.decoder(dec_in)
        return recon


class LitAutoencoder(pl.LightningModule):
    def __init__(self, lr=1e-3, base_channels=16, latent_dim=256, thresh=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvAutoencoder(1, base_channels, latent_dim)
        self.criterion = nn.MSELoss()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.thresh = thresh

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2)[:,0:1]
        recon, _ = self(x)
        loss = self.criterion(recon, x)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        if batch_idx % 200 == 0:
            metrics = self.compute_metrics(recon, x)
            self.log_dict({f'train_{k}':v for k,v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True)
            self.plot_images(x, recon, batch_idx, tag='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['vil'].permute(0,3,1,2)[:,0:1]
        recon, _ = self(x)
        loss = self.criterion(recon, x)
        metrics = self.compute_metrics(recon, x)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict({f'val_{k}':v for k,v in metrics.items()}, on_epoch=True, on_step=True, prog_bar=True)
        if batch_idx % 200 == 0:
            self.plot_images(x, recon, batch_idx, tag='val')

    def compute_metrics(self, recon, x):
        mse = nn.functional.mse_loss(recon, x)
        ssim = self.val_ssim(recon, x)
        psnr = self.val_psnr(recon, x)
        pred_bin = recon > self.thresh
        true_bin = x > self.thresh
        tp = (pred_bin & true_bin).sum(dim=[1,2,3]).float()
        fp = (pred_bin & ~true_bin).sum(dim=[1,2,3]).float()
        fn = (~pred_bin & true_bin).sum(dim=[1,2,3]).float()
        csi = torch.mean(tp / (tp + fp + fn + 1e-8))
        return {'mse': mse, 'ssim': ssim, 'psnr': psnr, 'csi': csi}

    def plot_images(self, x, recon, batch_idx, tag=''):
        cmap, norm = SEVIRLightningDataModule.vil_cmap()
        x = (x*255).clamp(0,255).to(torch.uint8)
        recon = (recon*255).clamp(0,255).to(torch.uint8)
        orig = x[0,0].cpu()
        pred = recon[0,0].cpu()
        fig, axes = plt.subplots(1,2,figsize=(8,4))
        axes[0].imshow(orig, cmap=cmap, norm=norm); axes[0].axis('off'); axes[0].set_title('Orig')
        axes[1].imshow(pred, cmap=cmap, norm=norm); axes[1].axis('off'); axes[1].set_title('Recon')
        plt.tight_layout() ; plt.savefig(f'plots/{tag}_batch_{batch_idx}.png') ; plt.close(fig)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'monitor': 'val_loss'}}

    def scheduler_step(self, scheduler, metric, optimizer_idx):
        scheduler.step()


if __name__ == '__main__':
    dm = SEVIRLightningDataModule(); dm.prepare_data(); dm.setup()
    model = LitAutoencoder(lr=1e-4)
    logger = TensorBoardLogger('tb_logs', name='autoencoder')
    trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=1, logger=logger, limit_val_batches=200)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
