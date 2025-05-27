import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from tqdm import tqdm
from diffusers import AutoencoderKL
from fno_dataset import *
from datetime import datetime

# Configuration
IMG_CHANNELS = 1
IMG_SIZE = 384
LATENT_CHANNELS = 4
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
KL_WEIGHT = 1e-6
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "/home/vatsal/NWM/fno_sevir/vae_weights"

def create_vae_model():
    """Create and return VAE model"""
    model = AutoencoderKL(
        in_channels=IMG_CHANNELS,
        out_channels=IMG_CHANNELS,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 64, 128, 256),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=LATENT_CHANNELS,
        norm_num_groups=32,
        sample_size=IMG_SIZE,
        scaling_factor=1.0,
        force_upcast=True,
        mid_block_add_attention=True
    )
    return model.to(DEVICE)

vil_map, vil_norm = vil_cmap()

def plot_reconstruction_comparison(original, reconstructed, epoch, batch_idx, save_wandb=True, vil_colormap=vil_map, vil_norm=vil_norm):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(f'VAE Reconstruction - Epoch {epoch+1}, Batch {batch_idx}')
    
    orig_np = original[0, 0].detach().cpu().numpy() * 255
    recon_np = reconstructed[0, 0].detach().cpu().numpy() * 255
    orig_np = np.clip(orig_np, 0, 255).astype(np.uint8)
    recon_np = np.clip(recon_np, 0, 255).astype(np.uint8)
    
    axes[0].imshow(orig_np, cmap=vil_colormap, norm=vil_norm)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(recon_np, cmap=vil_colormap, norm=vil_norm)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_wandb:
        wandb.log({f"reconstruction_epoch_{epoch}_batch_{batch_idx}": wandb.Image(fig)})
    plt.savefig(os.path.join("vae_plots", f"reconstruction_epoch_{epoch+1}_batch_{batch_idx}.png"), dpi=300)
    plt.close(fig)

def train_vae(model, train_loader, val_loader, optimizer, num_epochs):
    """Train VAE model"""
    wandb.init(project="vae-training", config={
        "learning_rate": LEARNING_RATE,
        "epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "kl_weight": KL_WEIGHT
    })
    
    model.train()
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    for epoch in range(1, num_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue
                
            images = batch_data["vil"].permute(0, 3, 1, 2).to(DEVICE)
            if batch_idx == 0:
                print(images.shape)

            # Forward pass
            encoder_output = model.encode(images)
            latents = encoder_output.latent_dist.sample()
            decoder_output = model.decode(latents)
            reconstructed = decoder_output.sample
            
            # Calculate losses
            recon_loss = F.mse_loss(reconstructed, images)
            kl_loss = encoder_output.latent_dist.kl().mean()
            total_loss = recon_loss + KL_WEIGHT * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}"
            })
            
        
        # Log epoch metrics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_recon = epoch_recon_loss / num_batches if num_batches > 0 else 0
        avg_kl = epoch_kl_loss / num_batches if num_batches > 0 else 0
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_recon_loss": avg_recon,
            "train_kl_loss": avg_kl
        })
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
        
        save_file = os.path.join(SAVE_PATH, f"vae_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_file)
        print(f"Model saved: {save_file}")
        
        # Validation
        validate_vae(model, val_loader, epoch)

def validate_vae(model, val_loader, epoch):
    """Validate VAE model"""
    model.eval()
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if batch_data is None:
                continue
                
            images = batch_data["vil"].permute(0, 3, 1, 2).to(DEVICE)
            
            encoder_output = model.encode(images)
            latents = encoder_output.latent_dist.sample()
            reconstructed = model.decode(latents).sample
            
            recon_loss = F.mse_loss(reconstructed, images)
            kl_loss = encoder_output.latent_dist.kl().mean()
            total_loss = recon_loss + KL_WEIGHT * kl_loss
            
            val_loss += total_loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:  # Plot first validation batch
                plot_reconstruction_comparison(images, reconstructed, epoch, batch_idx, save_wandb=True)
            if batch_idx > 300:
                break
                
    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
    wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    model.train()

def run_inference(model, test_loader, num_samples=5):
    """Run inference on test data"""
    model.eval()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            if idx >= num_samples or batch_data is None:
                break
                
            images = batch_data["vil"].permute(0, 3, 1, 2).to(DEVICE)#[B, 1, 384, 384]
            encoder_output = model.encode(images)
            latents = encoder_output.latent_dist.mean  # Use mean for inference
            reconstructed = model.decode(latents).sample
            
            plot_reconstruction_comparison(images, reconstructed, 0, idx, save_wandb=False)
            if idx > 10:
                break

def main():
    train_loader = SEVIRDataLoader(
        start_date=datetime(2017, 1, 1), 
        end_date=datetime(2019, 1, 1), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        seq_len=1, 
        stride=20, 
        rescale_method='01', 
        preprocess=True
    )
    
    val_loader = SEVIRDataLoader(
        start_date=datetime(2019, 5, 1), 
        end_date=datetime(2019, 7, 1), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        seq_len=1, 
        stride=20, 
        rescale_method='01', 
        preprocess=True
    )
    
    test_loader = SEVIRDataLoader(
        start_date=datetime(2019, 5, 1), 
        end_date=datetime(2019, 7, 1), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        seq_len=1, 
        stride=20, 
        rescale_method='01', 
        preprocess=True
    )
    for loader in [train_loader, val_loader, test_loader]:
        print(f"Dataset size: {len(loader)}")
        for sample in loader:
            vil = sample['vil']
            print(f"Sample shape: {vil.shape}, dtype: {vil.dtype}. Maximum value: {vil.max().item()}, Minimum value: {vil.min().item()}")
            break
    
    vae_model = create_vae_model()
    vae_model.load_state_dict(torch.load("/home/vatsal/NWM/fno_sevir/vae_weights/vae_epoch_1.pth", weights_only=False))
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print("Starting VAE training...")
    if len(train_loader) > 0:
        train_vae(vae_model, train_loader, val_loader, optimizer, NUM_EPOCHS)
    else:
        print("Training dataset is empty!")
        return
    
    # Inference
    print("Running inference...")
    run_inference(vae_model, test_loader)
    
    wandb.finish()

if __name__ == "__main__":
    main()