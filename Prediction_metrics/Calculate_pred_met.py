import os
import numpy as np
import csv
from .Prediction_metrics import *
import matplotlib.pyplot as plt
import torch

# Code by gpt
def calculate_predictionmetric_scores(dir, number_test_seq, seq_length, csv_file_path):
    pred_file = sorted([f for f in os.listdir(dir) if f.endswith("_ypred.npy")], 
                       key = lambda x: int(x.split("_")[0].replace("Sample", "")))
    gt_file = sorted([f for f in os.listdir(dir) if f.endswith("_ytest.npy")],
                     key = lambda x: int(x.split("_")[0].replace("Sample", "")))
    xtest_file = sorted([f for f in os.listdir(dir) if f.endswith("_xtest.npy")],
                        key = lambda x: int(x.split("_")[0].replace("Sample", "")))
    
    
    # Creating CSV headers
    headers = []
    for n in range(seq_length):
        headers.append(f"PCC(Channel{n+1:02d})")
        headers.append(f"RMSE(Channel{n+1:02d})")
    headers.append("PCC(Avg)")
    headers.append("RMSE(Avg)")
    
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Writing the header
        writer.writerow(["Sample/Persistence"] + headers)  

        for i in range(number_test_seq):
            persistence_add = os.path.join(dir, xtest_file[i])
            gt_add = os.path.join(dir, gt_file[i])
            pred_add = os.path.join(dir, pred_file[i])

            ground_truth = np.load(gt_add)[0]  # Shape (16, 480, 480)
            predicted = np.load(pred_add)[0]  # Shape (16, 480, 480)
            persistence = np.load(persistence_add)[0, -1, :, :]  # Shape (480, 480)

            sample_metrics = [f"Sample({i+1})"]
            persistence_metrics = [f"Persistence({i+1})"]
            sample_metrics1 = []
            sample_metrics2 = []
            persistence_metrics1 = []
            persistence_metrics2 = []

            for index in range(ground_truth.shape[0]):
                gt = ground_truth[index, :, :]
                pred = predicted[index, :, :]

                sample_metrics.append(PCC(gt, pred))
                sample_metrics.append(RMSE(gt, pred))
                sample_metrics1.append(PCC(gt, pred))
                sample_metrics2.append(RMSE(gt, pred))
                
                persistence_metrics.append(PCC(gt, persistence))
                persistence_metrics.append(RMSE(gt, persistence))
                persistence_metrics1.append(PCC(gt, persistence))
                persistence_metrics2.append(RMSE(gt, persistence))
            
            avg_pcc_pred =  np.mean([float(val) for val in sample_metrics1[1:]])
            avg_rmse_pred = np.mean([float(val) for val in sample_metrics2[1:]])

            avg_pcc_per = np.mean([float(val) for val in persistence_metrics1[1:]])
            avg_rmse_per = np.mean([float(val) for val in persistence_metrics2[1:]])

            sample_metrics.append(avg_pcc_pred)
            sample_metrics.append(avg_rmse_pred)

            persistence_metrics.append(avg_pcc_per)
            persistence_metrics.append(avg_rmse_per)

            writer.writerow(sample_metrics)
            writer.writerow(persistence_metrics)
    
    print(f"Metrics saved to {csv_file_path}")

def plot_predictionmetric_scores(dir, number_test_seq, seq_length):
    pred_file = sorted([f for f in os.listdir(dir) if f.endswith("_ypred.npy")], 
                       key = lambda x: int(x.split("_")[0].replace("Sample", "")))
    gt_file = sorted([f for f in os.listdir(dir) if f.endswith("_ytest.npy")],
                     key = lambda x: int(x.split("_")[0].replace("Sample", "")))
    xtest_file = sorted([f for f in os.listdir(dir) if f.endswith("_xtest.npy")],
                        key = lambda x: int(x.split("_")[0].replace("Sample", "")))
    
    sample_pcc = np.zeros(seq_length)
    sample_rmse = np.zeros(seq_length)
    persistence_pcc = np.zeros(seq_length)
    persistence_rmse = np.zeros(seq_length)
    
    sample_pcc_std = np.zeros(seq_length)
    sample_rmse_std = np.zeros(seq_length)
    persistence_pcc_std = np.zeros(seq_length)
    persistence_rmse_std = np.zeros(seq_length)
    
    all_sample_pcc = []
    all_sample_rmse = []
    all_persistence_pcc = []
    all_persistence_rmse = []

    dict = {}

    for i in range(number_test_seq):
        persistence_add = os.path.join(dir, xtest_file[i])
        gt_add = os.path.join(dir, gt_file[i])
        pred_add = os.path.join(dir, pred_file[i])

        ground_truth = np.load(gt_add)[0]  # Shape (16, 480, 480)
        predicted = np.load(pred_add)[0]  # Shape (16, 480, 480)
        persistence = np.load(persistence_add)[0, -1, :, :]  # Shape (480, 480)

        sample_metrics_pcc = []
        sample_metrics_rmse = []
        persistence_metrics_pcc = []
        persistence_metrics_rmse = []

        
        for index in range(ground_truth.shape[0]):
            gt = ground_truth[index, :, :]
            pred = predicted[index, :, :]

            sample_metrics_pcc.append(PCC(gt, pred))
            sample_metrics_rmse.append(RMSE(gt, pred))

            persistence_metrics_pcc.append(PCC(gt, persistence))
            persistence_metrics_rmse.append(RMSE(gt, persistence))
        
        all_sample_pcc.append(sample_metrics_pcc)
        all_sample_rmse.append(sample_metrics_rmse)
        all_persistence_pcc.append(persistence_metrics_pcc)
        all_persistence_rmse.append(persistence_metrics_rmse)
    
    all_sample_pcc = np.array(all_sample_pcc)
    all_sample_rmse = np.array(all_sample_rmse)
    all_persistence_pcc = np.array(all_persistence_pcc)
    all_persistence_rmse = np.array(all_persistence_rmse)
    
    sample_pcc = np.mean(all_sample_pcc, axis=0)
    sample_rmse = np.mean(all_sample_rmse, axis=0)
    persistence_pcc = np.mean(all_persistence_pcc, axis=0)
    persistence_rmse = np.mean(all_persistence_rmse, axis=0)
    
    sample_pcc_std = np.std(all_sample_pcc, axis=0)
    sample_rmse_std = np.std(all_sample_rmse, axis=0)
    persistence_pcc_std = np.std(all_persistence_pcc, axis=0)
    persistence_rmse_std = np.std(all_persistence_rmse, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(20, 20))

    x_axis = np.arange(1, seq_length + 1)

    axes[0].plot(x_axis, sample_pcc, marker='o', linestyle = '-', color= 'b', label = 'Sample PCC')
    axes[0].fill_between(x_axis, sample_pcc - sample_pcc_std, sample_pcc + sample_pcc_std, color = 'b', alpha=0.2)
    axes[0].plot(x_axis, persistence_pcc, marker = 's', linestyle = '--', color= 'r', label = 'Persistence PCC')
    axes[0].fill_between(x_axis, persistence_pcc- persistence_pcc_std, persistence_pcc+persistence_pcc_std, color = 'r', alpha=0.2)
    axes[0].set_title("Pearson Correlation Coefficient (PCC)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("PCC")
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(x_axis, sample_rmse, marker='o', linestyle='-', color='b', label='Sample RMSE')
    axes[1].fill_between(x_axis, sample_rmse - sample_rmse_std, sample_rmse + sample_rmse_std, color = 'b', alpha=0.2)
    axes[1].plot(x_axis, persistence_rmse, marker='s', linestyle='--', color='r', label='Persistence RMSE')
    axes[1].fill_between(x_axis, persistence_rmse- persistence_rmse_std, persistence_rmse+persistence_rmse_std, color = 'r', alpha=0.2)
    axes[1].set_title("Root Mean Square Error (RMSE)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    axes[1].grid()

    parent_path = os.path.dirname(dir)
    save_path1 = os.path.join(parent_path, f"Prediction_scores_plot")
    if not os.path.exists(save_path1):  # Corrected syntax
        os.makedirs(save_path1, exist_ok=True)  # `makedirs` ensures parent directories exist
    
    folder_name = os.path.basename(os.path.normpath(dir))
    save_path2 = os.path.join(save_path1, folder_name)
    if not os.path.exists(save_path2):  # Corrected condition
        os.makedirs(save_path2, exist_ok=True)

    
    save_file = os.path.join(save_path2, f"Plot_PCC_and_RMSE.png")
    


    fig.savefig(save_file)
    

    plt.close(fig)


    print(f"Saved: {save_file}")


    


# types = ["Red_mse_"]

# learning_rate = [0.001, 0.0001]

# activation = ["relu", "selu"]

# batch_size = [3,10, 20]

# root_dir = "/home/vatsal/Lion/All_Unet (less dataset experiment results)/Predictions unet"
# for type in types:
#     for lr in learning_rate:
#         for ac in activation:
#             for bs in batch_size:
#                 dir_add = os.path.join(root_dir, f"{type}{lr}_{ac}_{bs}")
#                 csv_file_path = os.path.join("/home/vatsal/MOSDAC/predictions/", f"{type}{lr}_{ac}_{bs}")

#                 if os.path.exists(dir_add):
#                     plot_predictionmetric_scores(dir_add,30,16)
#                 else:
#                     print("path doesnt exist")
#                     print(dir_add)




def plot_predictionmetric_scores2(plot_name, x_test,y_test,y_pred, number_test_seq, seq_length):
    # expects shape (batch_size, timesteps, height, width)


    sample_pcc = np.zeros(seq_length)
    sample_rmse = np.zeros(seq_length)
    persistence_pcc = np.zeros(seq_length)
    persistence_rmse = np.zeros(seq_length)
    
    sample_pcc_std = np.zeros(seq_length)
    sample_rmse_std = np.zeros(seq_length)
    persistence_pcc_std = np.zeros(seq_length)
    persistence_rmse_std = np.zeros(seq_length)
    
    all_sample_pcc = []
    all_sample_rmse = []
    all_persistence_pcc = []
    all_persistence_rmse = []

    
    for i in range(number_test_seq):

        
        x_test_ = x_test[i]
        y_test_ = y_test[i]
        y_pred_ = y_pred[i]

        
        
        persistence = x_test_[0,-1,:,:]
        ground_truth = y_test_[0]
        predicted = y_pred_[0]

        sample_metrics_pcc = []
        sample_metrics_rmse = []
        persistence_metrics_pcc = []
        persistence_metrics_rmse = []

        for index in range(ground_truth.shape[0]):
            gt = ground_truth[index, :, :]
            pred = predicted[index, :, :]

            sample_metrics_pcc.append(PCC(gt, pred))
            sample_metrics_rmse.append(RMSE(gt, pred))

            persistence_metrics_pcc.append(PCC(gt, persistence))
            persistence_metrics_rmse.append(RMSE(gt, persistence))
        
        all_sample_pcc.append(sample_metrics_pcc)
        all_sample_rmse.append(sample_metrics_rmse)
        all_persistence_pcc.append(persistence_metrics_pcc)
        all_persistence_rmse.append(persistence_metrics_rmse)

    all_sample_pcc = np.array(all_sample_pcc)
    all_sample_rmse = np.array(all_sample_rmse)
    all_persistence_pcc = np.array(all_persistence_pcc)
    all_persistence_rmse = np.array(all_persistence_rmse)

    sample_pcc = np.mean(all_sample_pcc, axis=0)
    sample_rmse = np.mean(all_sample_rmse, axis=0)
    persistence_pcc = np.mean(all_persistence_pcc, axis=0)
    persistence_rmse = np.mean(all_persistence_rmse, axis=0)
    
    sample_pcc_std = np.std(all_sample_pcc, axis=0)
    sample_rmse_std = np.std(all_sample_rmse, axis=0)
    persistence_pcc_std = np.std(all_persistence_pcc, axis=0)
    persistence_rmse_std = np.std(all_persistence_rmse, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(20, 20))

    x_axis = np.arange(1, seq_length + 1)

    axes[0].plot(x_axis, sample_pcc, marker='o', linestyle = '-', color= 'b', label = 'Sample PCC')
    axes[0].fill_between(x_axis, sample_pcc - sample_pcc_std, sample_pcc + sample_pcc_std, color = 'b', alpha=0.2)
    axes[0].plot(x_axis, persistence_pcc, marker = 's', linestyle = '--', color= 'r', label = 'Persistence PCC')
    axes[0].fill_between(x_axis, persistence_pcc- persistence_pcc_std, persistence_pcc+persistence_pcc_std, color = 'r', alpha=0.2)
    axes[0].set_title("Pearson Correlation Coefficient (PCC)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("PCC")
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(x_axis, sample_rmse, marker='o', linestyle='-', color='b', label='Sample RMSE')
    axes[1].fill_between(x_axis, sample_rmse - sample_rmse_std, sample_rmse + sample_rmse_std, color = 'b', alpha=0.2)
    axes[1].plot(x_axis, persistence_rmse, marker='s', linestyle='--', color='r', label='Persistence RMSE')
    axes[1].fill_between(x_axis, persistence_rmse- persistence_rmse_std, persistence_rmse+persistence_rmse_std, color = 'r', alpha=0.2)
    axes[1].set_title("Root Mean Square Error (RMSE)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    axes[1].grid()


    dir = "/home/vatsal/MOSDAC/predictions/prediction_scores_plot"
    save_path1 = os.path.join(dir, f"{plot_name}")
    if not os.path.exists(save_path1):  # Corrected syntax
        os.makedirs(save_path1, exist_ok=True)  # `makedirs` ensures parent directories exist
    
    

    
    save_file = os.path.join(save_path1, f"Plot_PCC_and_RMSE.png")
    


    fig.savefig(save_file)
    

    plt.close(fig)


    print(f"Saved: {save_file}")


def cal_batch_avg_pcc(y, y_pred, seq_length):
    bs = y.shape[0]
    y = y.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    pccs = []
    for i in range(bs):
        pcc_seq = []
        for j in range(seq_length):
            pcc_seq.append(PCC(y[i][j], y_pred[i][j]))
        pcc_seq = np.array(pcc_seq)
        # print(pcc_seq.shape)
        mean = np.mean(pcc_seq, axis = 0)
        pccs.append(mean)
        # print(mean.shape)
    
    pccs = np.array(pccs)
    # print(f"Pccs shape : {pccs}")
    avg_pcc = np.mean(pccs, axis = 0)
    # print(f"Avg Pccs shape : {avg_pcc}")
    return avg_pcc

def cal_batch_avg_psnr(y, y_pred, seq_length):
    bs = y.shape[0]
    y = y.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    psnrs = []
    for i in range(bs):
        psnr_seq = []
        for j in range(seq_length):
       
            psnr_seq.append(PSNR(y[i][j], y_pred[i][j]))
        psnr_seq = np.array(psnr_seq)
        # print(pcc_seq.shape)
        mean = np.mean(psnr_seq, axis = 0)
        psnrs.append(mean)
        # print(mean.shape)
    
    psnrs = np.array(psnrs)
    # print(f"Pccs shape : {pccs}")
    avg_psnr = np.mean(psnrs, axis = 0)
    # print(f"Avg Pccs shape : {avg_pcc}")
    return avg_psnr

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


def pcc_loss_batch(y_pred, y_true, eps=1e-8):
    """
    y_pred, y_true: (B, H, W) tensors
    Computes 1 - Pearson correlation coefficient for each sample in the batch
    Returns: scalar loss
    """
    B = y_true.size(0)

    # Flatten H x W to a vector for each sample
    y_pred_flat = y_pred.view(B, -1)
    y_true_flat = y_true.view(B, -1)

    # Subtract means
    y_pred_mean = y_pred_flat.mean(dim=1, keepdim=True)
    y_true_mean = y_true_flat.mean(dim=1, keepdim=True)

    x = y_pred_flat - y_pred_mean
    y = y_true_flat - y_true_mean

    # Compute covariance and standard deviations
    cov = (x * y).sum(dim=1)
    x_std = torch.sqrt((x ** 2).sum(dim=1) + eps)
    y_std = torch.sqrt((y ** 2).sum(dim=1) + eps)

    pcc = cov / (x_std * y_std)

    # Final loss: 1 - mean PCC across the batch
    return 1 - pcc.mean()

    