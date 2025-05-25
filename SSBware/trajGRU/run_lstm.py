import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import os
import wandb
wandb.disabled = True
import numpy as np
from Rainy_Dataset import *
from Calculate_pred_met import *
import argparse
from neuralop.models import FNO
from torchmetrics.functional import structural_similarity_index_measure as ssim

device = "cuda"
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run the extreme rainfall event detection model.")
    
    # Dataset parameters
    parser.add_argument('--sequence_length', type=int, default=10, help='Number of sequential files per data point.')
    
    # Model parameters
    parser.add_argument('--input_channel', type=int, default=10, help='Number of channels to pass as the input')
    parser.add_argument('--output_channel', type=int, default=10, help='Number of channels in the output')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--optimizer', type=str, default="SGD", help='optimizer.')
    parser.add_argument('--modes', type=int, default=64, help='Number of modes for FNO.')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels')
    parser.add_argument('--alpha', type= float, default= 0.7, help= " Indicates the dominance of a particular loss")
    # Misc
    parser.add_argument('--train_test_split', type=float, default=0.9, help='Fraction of data used for training (rest for testing).')

    # parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    
    return parser.parse_args()

def main(args):
    print("Using", device)
    data_dir = '/home/vatsal/MOSDAC/train_test/full_dataset/'
    train_dataset, test_dataset, val_dataset = rainy_dataset(data_dir,args.sequence_length,args.sequence_length)

    train_loader = create_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_loader(val_dataset, batch_size=args.batch_size)
    test_loader = create_loader(test_dataset, batch_size=1)

    model = FNO(n_modes=(args.modes, args.modes), hidden_channels=args.hidden_channels, in_channels=args.input_channel, out_channels=args.output_channel)
    print(args.modes, args.hidden_channels, args.input_channel, args.output_channel)
    model = model
	#model = torch.compile(model)
    # print(summary(model, (args.batch_size,10, 480, 480)))

    mse_loss = nn.MSELoss()
    huber_loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum= 0.9)

    # summary(model, input_size=(args.batch_size, args.input_channel, 480, 480))
    # exit()

    # early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    #wandb.init(project=f"FNO_bestofsweep_PCC_MSE_with prevdays")

    print("Training")

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        avg_train_loss = []
        for x, y in train_loader:
            print(x.shape)
            print(y.shape)
            pred = model(x)  
            loss = args.alpha*pcc_loss_batch(pred, y) + (1-args.alpha) * mse_loss(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_train_loss.append(loss.item())

        train_losses = (np.mean(avg_train_loss))

        model.eval()
        avg_val_loss = []
        sample_metrics_pcc = []
        sample_metrics_ssim = []
        sample_metrics_psnr = []
        sample_metrics_acc = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                # for channels
                x_val, y_val = x_val.to(device), y_val.to(device)
                # print(x_val.shape)
                # print(y_val.shape)
                val_pred = model(x_val)
                val_loss = args.alpha*pcc_loss_batch(val_pred, y_val) + (1-args.alpha) * mse_loss(val_pred, y_val)
                avg_val_loss.append(val_loss.item())
                sample_metrics_pcc.append(cal_batch_avg_pcc(y_val, val_pred, args.sequence_length))
                sample_metrics_ssim.append(ssim(y_val, val_pred, data_range = 50.0).cpu().numpy())
                sample_metrics_psnr.append(cal_batch_avg_psnr(y_val, val_pred, args.sequence_length))
                sample_metrics_acc.append(compute_batch_avg_acc(y_val, val_pred))
                                        
        val_losses = (np.mean(avg_val_loss))

        sample_metrics_pcc = np.array(sample_metrics_pcc)

        sample_pcc = np.mean(sample_metrics_pcc)
        
        sample_ssim = np.mean(sample_metrics_ssim)

        sample_psnr = np.mean(sample_metrics_psnr)

        sample_acc = np.mean(sample_metrics_acc)

        if not os.path.exists('/home/vatsal/Supreme/Mosdac_model_weights/PCC_MSE_withprevdays/'):
            os.makedirs('/home/vatsal/Supreme/Mosdac_model_weights/PCC_MSE_withprevdays/', exist_ok=True)

        if val_losses < best_val_loss:
            best_val_loss = val_losses
            torch.save(model.state_dict(), f'/home/vatsal/Supreme/Mosdac_model_weights/PCC_MSE_withprevdays/Best_FNO_{args.optimizer}_{args.learning_rate}_{args.batch_size}_{args.modes}_{args.hidden_channels}.pt')  # save best checkpoint
            print(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")
            print(f"Model saved, val_loss: {val_loss}")

        #wandb.log({"Train Loss": train_losses, "PCC": sample_pcc, "Validation_loss": val_losses, "SSIM": sample_ssim, "PSNR": sample_psnr, "ACC": sample_acc})
        print(f"Epoch {epoch+1}, Train Loss: {train_losses:.6f}, Val Loss: {val_losses:.6f}, SSIM: {sample_ssim}, PSNR: {sample_psnr}, ACC: {sample_acc}")

    #wandb.finish()

    model = FNO(n_modes=(args.modes, args.modes), hidden_channels=args.hidden_channels, in_channels=args.input_channel, out_channels=args.output_channel)

    # Find the best dataset

    state_dict = torch.load(f'/home/vatsal/Supreme/Mosdac_model_weights/PCC_MSE_withprevdays/Best_FNO_{args.optimizer}_{args.learning_rate}_{args.batch_size}_{args.modes}_{args.hidden_channels}.pt', weights_only=False)
    print("Retreived model")
    model.load_state_dict(state_dict)

    model = model.to(device)
    torch.compile(model)

    model.eval()

    print("Test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    x_test_lis = []
    y_test_lis = []
    y_pred_lis = []

    with torch.no_grad():  # Disable gradient computation for inference
        for i, (x_test, y_test) in enumerate(test_loader):
        
            x_test = x_test.to(device)  # Move input to GPU
            pred = model(x_test)  # Forward pass
            
            # Move prediction back to CPU and convert to numpy

            x_test, pred = x_test.cpu().numpy(), pred.cpu().numpy() 
            pred = np.where((pred<0), 0, pred) 
            y_test = y_test.numpy()
            # print(f"Predicted shape: {pred.shape}")
            # print(f"Test input shape:{x_test.shape}")
            # print(f"Test o/p shape:{y_test.shape}")

            x_test_lis.append(x_test)
            y_test_lis.append(y_test)
            y_pred_lis.append(pred)


    plot_predictionmetric_scores2(f"/home/vatsal/MOSDAC/predictions2/FNO_2_PCC_MSE_withprevdays/FNO_{args.optimizer}_{args.learning_rate}_{args.batch_size}_{args.modes}_{args.hidden_channels}", x_test_lis, y_test_lis, y_pred_lis, len(x_test_lis), args.sequence_length)

if __name__ == "__main__":
    args = parse_args()
    main(args)
