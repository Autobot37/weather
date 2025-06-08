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