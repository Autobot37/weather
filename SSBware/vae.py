import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import glob 

from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput 

IMG_CHANNELS = 1
IMG_SIZE = 512
LATENT_CHANNELS = 4
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 
KL_WEIGHT = 1e-6 # Weight for the KL divergence loss term
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "weights"

vae = AutoencoderKL(
    in_channels=IMG_CHANNELS,
    out_channels=IMG_CHANNELS,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(32, 64, 64, 128), 
    layers_per_block=2,
    act_fn="silu",
    latent_channels=LATENT_CHANNELS,
    norm_num_groups=32,
    sample_size=IMG_SIZE,
    scaling_factor=1.0, # For a new VAE, 1.0 is a safe start. Can be tuned/calculated later.
    force_upcast=True, # good for stability
    mid_block_add_attention=True 
)
vae.to(DEVICE)

optimizer = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE)

def train_vae(model, dataloader, optimizer, num_epochs, kl_weight, device, save_path):
    if len(dataloader.dataset) == 0:
        print("Skipping training as dataset is empty.")
        return

    model.train()
    for epoch in range(2, num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss_epoch = 0
        total_recon_loss_epoch = 0
        total_kl_loss_epoch = 0

        out = None

        for batch_idx, images in enumerate(progress_bar):
            if images is None:
                print(f"Skipping empty batch at index {batch_idx}")
                continue
            out = images
            images = images["input"]
            images = images.to(device)

            # The .encode() method returns an AutoencoderKLOutput object
            # which has a .latent_dist attribute (a DiagonalGaussianDistribution)
            encoder_output = model.encode(images)
            latent_dist = encoder_output.latent_dist

            # Sample from the latent distribution (reparameterization trick is applied inside .sample())
            latents = latent_dist.sample()
            # Note: The scaling_factor is NOT applied to these latents by default by .sample()
            # It's usually applied externally when these latents are fed to another model like a diffusion U-Net.
            # If you want to apply it here for some reason: latents = latents * model.config.scaling_factor

            # Decode
            # The .decode() method expects unscaled latents.
            # It returns a DecoderOutput object with a .sample attribute for the reconstructed images.
            decoder_output = model.decode(latents)
            reconstructed_images = decoder_output.sample

            # Calculate losses
            recon_loss = F.mse_loss(reconstructed_images, images, reduction="mean")
            kl_loss = latent_dist.kl().mean() # .kl() gives per-latent KL, then average over batch

            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            total_recon_loss_epoch += recon_loss.item()
            total_kl_loss_epoch += kl_loss.item()

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}"
            })

        avg_loss = total_loss_epoch / len(dataloader)
        avg_recon_loss = total_recon_loss_epoch / len(dataloader)
        avg_kl_loss = total_kl_loss_epoch / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg KL Loss: {avg_kl_loss:.4f}")

        sample = {
            "input": images[0].cpu(),
            "target": reconstructed_images[0].detach().cpu(),
            "lat": out["lat"][0],
            "lon": out["lon"][0],
        }
        plot_all_frames(sample, vmin=-30, vmax=70, label1="Input Frame", label2="Reconstructed Frame", save=True, name=f"vae epoch_{epoch+1}")

        save_path = os.path.join(save_path, f"vae_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

print("Starting VAE training...")
if len(train_loader.dataset) > 0:
        train_vae(vae, train_loader, optimizer, NUM_EPOCHS, KL_WEIGHT, DEVICE, MODEL_SAVE_PATH)
else:
    print(f"Cannot start training.")


def inference_with_vae(model, sample, output_path="vae_output.png"):
    model = model.to(DEVICE)
    model.eval()
    input_tensor = sample["input"].to(DEVICE) # is batched [B, 1, H, W] but we plot only first
    with torch.no_grad():
        latent_dist = model.encode(input_tensor).latent_dist
        latents = latent_dist.mean # or latent_dist.mean for Gaussian
        reconstructed_tensor = model.decode(latents).sample

    plot_all_frames(
        {"input": input_tensor[0].cpu(), "target": reconstructed_tensor[0].cpu(), "lat": sample["lat"][0], "lon": sample["lon"][0]},
        vmin=-30, vmax=70,
        label1="Input Frame",
        label2="Reconstructed Frame",
        save=False,
        name=output_path
    )
idx = 0
for sample in test_loader:
    inference_with_vae(vae, sample, output_path="vae_output.png")
    if idx > 5:
        break
    idx += 1