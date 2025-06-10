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

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
    
    @staticmethod
    def get_logger(model_name : str, run_name: str = None, save_dir: str = None) -> WandbLogger:
        if run_name is not None:
            return WandbLogger(
                project=model_name,
                name=run_name,
                save_dir=save_dir,
            )
        return WandbLogger(
                project= model_name,
                save_dir=save_dir,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement your forward pass in the subclass.")

    def _merge_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T, C, H, W) -> (B*T, C, H, W).
        """
        B, T, C, H, W = x.shape
        return x.reshape(B * T, C, H, W)

    def _psnr(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr_fn(preds, target, data_range=1.0).mean()

    def _ssim(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ssim_fn(preds, target, data_range=1.0).mean()

    def _csi(self, preds: torch.Tensor, target: torch.Tensor, csi_thresh = 0.5) -> torch.Tensor:
        p = (preds > csi_thresh).float()
        t = (target > csi_thresh).float()
        tp = (p * t).sum(dim=[1,2,3])
        fn = ((1 - p) * t).sum(dim=[1,2,3])
        fp = (p * (1 - t)).sum(dim=[1,2,3])
        return (tp / (tp + fn + fp + 1e-6)).mean()

    def _hss(self, preds: torch.Tensor, target: torch.Tensor, csi_thresh = 0.5) -> torch.Tensor:
        p  = (preds > csi_thresh).float()
        t  = (target > csi_thresh).float()
        tp = (p * t).sum()
        fn = ((1 - p) * t).sum()
        fp = (p * (1 - t)).sum()
        tn = ((1 - p) * (1 - t)).sum()
        num = 2 * (tp*tn - fn*fp)
        den = ( (tp+fn)*(fn+tn) + (tp+fp)*(fp+tn) ) + 1e-6
        return (num / den).clamp(-1, 1)

    def cal_CRPS(self, pred: torch.Tensor,target: torch.Tensor , type='avg', scale=4, mode='mean', eps=1e-10):
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
        return crps.item()

    def log_metrics(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        stage: str
    ):
        """
        preds, target: (B, T, C, H, W) in [0,1]
        logs: lpips ([-1,1]), psnr, ssim, csi, hss
        """
        p01 = self._merge_time(preds).clamp(0,1)
        t01 = self._merge_time(target).clamp(0,1)

        pm1 = (p01 * 2 - 1).clamp(-1,1)
        tm1 = (t01 * 2 - 1).clamp(-1,1)

        metrics: Dict[str, torch.Tensor] = {
            f"{stage}/psnr" : self._psnr(p01, t01),
            f"{stage}/ssim" : self._ssim(p01, t01),
            f"{stage}/csi"  : self._csi(p01, t01),
            f"{stage}/hss"  : self._hss(p01, t01),
        }
        for name, val in metrics.items():
            self.log(name, val, prog_bar=True, on_epoch=True, sync_dist=True, on_step=True)