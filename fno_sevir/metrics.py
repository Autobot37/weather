import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
def compute_batch_avg_acc(preds, targets):
    """
    Compute Anomaly Correlation Coefficient (ACC) for a batch of 2D images.

    Args:
        preds (Tensor): Predicted tensor of shape (B, H, W)
        targets (Tensor): Ground truth tensor of shape (B, H, W)

    Returns:
        Tensor: ACC per sample (shape: [B])
    """
    B = preds.size(0)
    preds = preds.view(B, -1)
    targets = targets.view(B, -1)

    pred_anom = preds - preds.mean(dim=1, keepdim=True)
    target_anom = targets - targets.mean(dim=1, keepdim=True)

    numerator = (pred_anom * target_anom).sum(dim=1)
    denominator = pred_anom.norm(dim=1) * target_anom.norm(dim=1) + 1e-8  # avoid div by zero
 
    acc = numerator / denominator
    avg_acc = torch.mean(acc, axis=0)
    return avg_acc.cpu().numpy()


def calculate_metrics(pred, true, x_input):
    """
    Calculate metrics for prediction vs ground truth and persistence baseline
    pred, true: shape (B, SEQ_LEN, H, W)
    x_input: shape (B, INPUT_SEQ_LEN, H, W) - input sequence for persistence baseline
    ## works with only batch 1
    """
    B, SEQ_LEN, H, W = pred.shape
    
    sample_pcc_values = []
    sample_ssim_values = []
    sample_psnr_values = []
    sample_rmse_values = []
    
    persistence_pcc_values = []
    persistence_ssim_values = []
    persistence_psnr_values = []
    persistence_rmse_values = []
    
    for b in range(B):
        # Persistence baseline: last frame of input sequence
        persistence_frame = x_input[b, -1]
        
        for t in range(SEQ_LEN):
            pred_frame = pred[b, t]
            true_frame = true[b, t]
            
            # Sample metrics
            pred_flat = pred_frame.flatten()
            true_flat = true_frame.flatten()
            pcc = np.corrcoef(pred_flat, true_flat)[0, 1]
            sample_pcc_values.append(pcc if not np.isnan(pcc) else 0)
            
            sample_ssim_values.append(ssim(true_frame, pred_frame, data_range=1.0))
            sample_psnr_values.append(psnr(true_frame, pred_frame, data_range=1.0))
            sample_rmse_values.append(np.sqrt(np.mean((pred_frame - true_frame) ** 2)))
            
            # Persistence metrics
            pers_flat = persistence_frame.flatten()
            pcc_pers = np.corrcoef(true_flat, pers_flat)[0, 1]
            persistence_pcc_values.append(pcc_pers if not np.isnan(pcc_pers) else 0)
            # print("persistence_frame shape", persistence_frame.shape)
            # print("true_frame shape", true_frame.shape)
            # print("pred_frame shape", pred_frame.shape)
            persistence_ssim_values.append(ssim(true_frame, persistence_frame, data_range=1.0))
            persistence_psnr_values.append(psnr(true_frame, persistence_frame, data_range=1.0))
            persistence_rmse_values.append(np.sqrt(np.mean((persistence_frame - true_frame) ** 2)))
    
    return sample_pcc_values, sample_ssim_values, sample_psnr_values, sample_rmse_values, \
           persistence_pcc_values, persistence_ssim_values, persistence_psnr_values, persistence_rmse_values

def get_metric_names():
    """
    Return list of metric names for logging
    """
    return [
        'sample_pcc', 'sample_ssim', 'sample_psnr', 'sample_rmse',
        'persistence_pcc', 'persistence_ssim', 'persistence_psnr', 'persistence_rmse'
    ]

def calculate_metrics_dict(pred, true, x_input):
    """
    Calculate metrics and return as dictionary for easy logging
    pred, true: shape (B, SEQ_LEN, H, W)
    """
    sample_pcc, sample_ssim, sample_psnr, sample_rmse, \
    persistence_pcc, persistence_ssim, persistence_psnr, persistence_rmse = calculate_metrics(pred, true, x_input)
    metrics_dict = {}
    for i in range(len(sample_pcc)):
        metrics_dict[f'sample_pcc_{i}'] = sample_pcc[i]
        metrics_dict[f'sample_ssim_{i}'] = sample_ssim[i]
        metrics_dict[f'sample_psnr_{i}'] = sample_psnr[i]
        metrics_dict[f'sample_rmse_{i}'] = sample_rmse[i]
        
        metrics_dict[f'persistence_pcc_{i}'] = persistence_pcc[i]
        metrics_dict[f'persistence_ssim_{i}'] = persistence_ssim[i]
        metrics_dict[f'persistence_psnr_{i}'] = persistence_psnr[i]
        metrics_dict[f'persistence_rmse_{i}'] = persistence_rmse[i]
    return metrics_dict