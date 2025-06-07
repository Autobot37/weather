import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from einops import rearrange
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------
# 3. Metrics Definitions
# -----------------------------
# Standard image metrics
lpips_fn = lpips.LPIPS(net="alex").to(device)
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Threshold-based metrics
@torch.no_grad()
def _threshold(tgt, pred, thr):
    return (tgt >= thr).int(), (pred >= thr).int()

@torch.no_grad()
def calc_hits_misses_fas(pred, target, thr, dims=(1,2,3)):
    t, p = _threshold(target, pred, thr)
    hits = torch.sum(t * p, dim=dims).int()
    misses = torch.sum(t * (1 - p), dim=dims).int()
    fas = torch.sum((1 - t) * p, dim=dims).int()
    cor = torch.sum((1 - t) * (1 - p), dim=dims).int()
    return hits, misses, fas, cor

@torch.no_grad()
def pod(hits, misses, fas, eps=1e-6):
    return hits.float() / (hits + misses + eps)

@torch.no_grad()
def sucr(hits, misses, fas, eps=1e-6):
    return hits.float() / (hits + fas + eps)

@torch.no_grad()
def csi(hits, misses, fas, eps=1e-6):
    return hits.float() / (hits + misses + fas + eps)

@torch.no_grad()
def bias(hits, misses, fas, eps=1e-6):
    b = (hits + fas).float() / (hits + misses + eps)
    return torch.pow(b / torch.log(torch.tensor(2.0, device=b.device)), 2.0)

@torch.no_grad()
def hss(hits, misses, fas, cor, eps=1e-6):
    num = 2 * (hits * cor - misses * fas).float()
    den = ((hits + misses) * (misses + cor) + (hits + fas) * (fas + cor) + eps)
    return num / den