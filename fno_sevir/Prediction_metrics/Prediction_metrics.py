import numpy as np
import torch.nn.functional as F
import torch

def PCC(target, prediction):
    # Convert tensors to numpy if needed
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    
    # Flatten the images
    target = target.ravel()
    prediction = prediction.ravel()

    # Ensure target and prediction have the same size
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction size don't match.")

    # Compute mean
    target_mean = np.mean(target)
    prediction_mean = np.mean(prediction)

    # Compute Pearson Correlation Coefficient
    target_centered = target - target_mean
    prediction_centered = prediction - prediction_mean

    numerator = np.sum(target_centered * prediction_centered)
    denominator = np.sqrt(np.sum(target_centered**2)) * np.sqrt(np.sum(prediction_centered**2))

    # Handle edge case where denominator is zero
    if denominator == 0:
        return 0  # Define PCC as 0 in case of no variance

    return numerator / denominator

def RMSE(target, prediction):
    # Convert tensors to numpy if needed
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    
    # Flatten the images
    target = target.ravel()
    prediction = prediction.ravel()

    if target.shape != prediction.shape:
        raise ValueError("Target and prediction size don't match.")
    
    # Count non-zero pixels for normalization
    len_target = (target != 0).sum()
    if len_target == 0:
        len_target = len(target)  # Fallback to total length if all zeros
    
    residual = target - prediction
    return np.sqrt(np.sum(residual**2) / len_target)

def compute_mse(img1, img2):
    # Convert tensors to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    return np.mean((img1 - img2) ** 2)

def PSNR(img1, img2, max_val=50):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr



