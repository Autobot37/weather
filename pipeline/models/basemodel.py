import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim_fn,
    peak_signal_noise_ratio          as psnr_fn,
)
from pytorch_lightning.loggers import WandbLogger
from typing import Any, Dict, Tuple
from omegaconf import OmegaConf
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from lpips import LPIPS as lpips
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class CodeLogger(Callback):
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        run = trainer.logger.experiment
        code_dir = Path(__file__).resolve().parents[1]  # pipeline/
        artifact = wandb.Artifact("pipeline_code", type="code")
        for f in code_dir.rglob("*"):
            if f.suffix in {".py", ".yaml", ".yml", ".ipynb"} and f.is_file():
                artifact.add_file(str(f), name=str(f.relative_to(code_dir)))
        run.log_artifact(artifact)

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lpips_fn = lpips('alex')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement your forward pass in the subclass.")

    def _merge_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T, C, H, W) -> (B*T, C, H, W).
        """
        B, T, C, H, W = x.shape
        return x.reshape(B * T, C, H, W)

    def _psnr(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return psnr_fn(preds, targets, data_range=1.0).mean()

    def _ssim(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ssim_fn(preds, targets, data_range=1.0).mean()

    def _csi(self, preds: torch.Tensor, targets: torch.Tensor, radius=1 , csi_thresh = 0.5) -> torch.Tensor:
        #input_shape : B*T,C,H,W
        if radius > 1:
            pool = nn.MaxPool2d(kernel_size= radius, stride=1)
            B,C,_, _= targets.shape
            csi_list = []
            for b in range(B):
                preds_ = pool(preds[b])
                target_ = pool(targets[b]) 
                p = (preds_ > csi_thresh).float()
                tgt_bin = (target_ > csi_thresh).float()

                tp = (p * tgt_bin).sum()
                fn = ((1 - p) * tgt_bin).sum()
                fp = (p * (1 - tgt_bin)).sum()
                csi_list.append((tp / (tp + fn + fp + 1e-6)).mean())
            return torch.stack(csi_list).mean()
        
        else: 
            p = (preds > csi_thresh).float()
            t = (targets > csi_thresh).float()
            tp = (p * t).sum(dim=[1,2,3])
            fn = ((1 - p) * t).sum(dim=[1,2,3])
            fp = (p * (1 - t)).sum(dim=[1,2,3])
            csi = (tp / (tp + fn + fp + 1e-6))
            return csi.mean()

    def _hss(self, preds: torch.Tensor, targets: torch.Tensor, csi_thresh = 0.75) -> torch.Tensor:
        p  = (preds > csi_thresh).float()
        t  = (targets > csi_thresh).float()
        tp = (p * t).sum()
        fn = ((1 - p) * t).sum()
        fp = (p * (1 - t)).sum()
        tn = ((1 - p) * (1 - t)).sum()
        num = 2 * (tp*tn - fn*fp)
        den = ( (tp+fn)*(fn+tn) + (tp+fp)*(fp+tn) ) + 1e-6
        return (num / den).clamp(-1, 1)

    def _CRPS(self, pred: torch.Tensor,targets: torch.Tensor , type='avg', scale=4, mode='mean', eps=1e-10):
        """
        gt: (b, t, c, h, w)
        pred: (b, n, t, c, h, w)
        """
        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'
        _normal_dist = torch.distributions.Normal(0, 1)
        _frac_sqrt_pi = 1 / torch.sqrt(np.pi)

        b, n, t, _, _, _ = pred.shape
        gt = rearrange(gt, 'b t c h w -> (b t) c h w')
        pred = rearrange(pred, 'b n t c h w -> (b n t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, scale, stride=scale)
            gt = F.avg_pool2d(gt, scale, stride=scale)
        elif type == 'max':
            pred = F.max_pool2d(pred, scale, stride=scale)
            gt = F.max_pool2d(gt, scale, stride=scale)
        else:
            gt = gt
            pred = pred
        gt = rearrange(gt, '(b t) c h w -> b t c h w', b=b)
        pred = rearrange(pred, '(b n t) c h w -> b n t c h w', b=b, n=n)

        pred_mean = torch.mean(pred, dim=1)
        pred_std = torch.std(pred, dim=1) if n > 1 else torch.zeros_like(pred_mean)
        normed_diff = (pred_mean - gt + eps) / (pred_std + eps)
        cdf = _normal_dist.cdf(normed_diff)
        pdf = _normal_dist.log_prob(normed_diff).exp()

        crps = (pred_std + eps) * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
        if mode == "mean":
            return torch.mean(crps).item()
        return crps

    def _lpips(self, preds, targets):
        """
        preds, targets: (B*T C, H, W) in [-1,1]
        """
        N = preds.shape[0]
        lpips_scores = []
        for i in range(N):
            score = self.lpips_fn(preds[i:i+1], targets[i:i+1])
            lpips_scores.append(score)
        return torch.stack(lpips_scores).mean()

    def _psd(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        [B, T, H, W][0, 1]
        """
        def rapsd(image: torch.Tensor, return_freq: bool = False) -> torch.Tensor:
            assert image.ndim == 2, "Input must be a 2D tensor (grayscale image)."

            H, W = image.shape
            f = torch.fft.fft2(image)
            fshift = torch.fft.fftshift(f)
            psd2D = torch.abs(fshift) ** 2

            y, x = torch.meshgrid(torch.arange(H, device=image.device),
                                torch.arange(W, device=image.device),
                                indexing='ij')
            y = y - H // 2
            x = x - W // 2
            r = torch.sqrt(x**2 + y**2).round().long()

            r_flat = r.view(-1)
            psd_flat = psd2D.view(-1)
            r_max = r_flat.max().item() + 1

            radial_psd = torch.zeros(r_max, dtype=psd2D.dtype, device=image.device)
            counts = torch.zeros(r_max, dtype=psd2D.dtype, device=image.device)

            radial_psd.scatter_add_(0, r_flat, psd_flat)
            counts.scatter_add_(0, r_flat, torch.ones_like(psd_flat))

            radial_psd = radial_psd / (counts + 1e-8)

            if return_freq:
                freqs = torch.arange(r_max, device=image.device) / (max(H, W) / 2)
                return radial_psd, freqs

            return radial_psd
        B, C, _, _ = gt.shape
        similarity_score = []

        for b in range(B):
            for c in range(C):
                rapsd_pred = rapsd(pred[b, c])
                rapsd_gt = rapsd(gt[b, c])
                sim = -torch.norm(rapsd_pred - rapsd_gt, p=2)
                similarity_score.append(sim)

        return torch.stack(similarity_score).mean()
    
    def log_plots(self, preds, targets, plot_fn, label):
        #[B T H W][0, 1]
        fig = plot_fn({"predictions": preds, "targets": targets}, label=label)
        self.logger.experiment.log({
            f"plots/{label}": wandb.Image(fig)
        })
        plt.close(fig)

    def log_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        stage: str
    ):
        """
        preds, targets: (B, T, C, H, W) in [0,1]
        logs: lpips ([-1,1]), psnr, ssim, csi, hss
        """
        p01 = self._merge_time(preds).clamp(0,1)
        t01 = self._merge_time(targets).clamp(0,1)

        pm1 = (p01 * 2 - 1).clamp(-1,1)
        tm1 = (t01 * 2 - 1).clamp(-1,1)

        ##radius 1, 4, 16
        ## t 0.5, 0.75
        metrics: Dict[str, torch.Tensor] = {
            f"{stage}/psnr" : self._psnr(p01, t01),
            f"{stage}/ssim" : self._ssim(p01, t01),
            f"{stage}/hss"  : self._hss(p01, t01),
            f"{stage}/lpips": self._lpips(pm1, tm1),
            f"{stage}/psd" : self._psd(p01, t01),
        }
        radius = [1, 4, 16]
        thresh = [0.5, 0.75]
        for r in radius:
            for t in thresh:
                metrics[f"{stage}/csi_radius_{r}_thresh_{t}"] = self._csi(p01, t01, radius=r, csi_thresh=t)

        for name, val in metrics.items():
            self.log(name, val, prog_bar=True, on_epoch=True, sync_dist=True, on_step=True)