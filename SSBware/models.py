## 19GB 
model = UNet2DModel(
    sample_size=512,  # the target image resolution
    in_channels=1 + 2,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 32, 64, 64, 128, 128),    
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
NUM_TRAIN_TIMESTEPS = 1000
noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
n_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
print("Using device:", device)
model = model.to(device)
torch.compile(model)
loss_fn = nn.MSELoss()
model.train()
print("Training model...")

opt = torch.optim.Adam(model.parameters(), lr=5e-4)
losses = []
from tqdm import tqdm
for epoch in range(n_epochs):
    for out in tqdm(train_loader):
        condition = out["input"].float() # [B, window_size, 512, 512]
        target = out["target"].float()  # [B, 1, 512, 512] 
        condition = condition.to(device) 
        target = target.to(device)
        
        noise = torch.randn_like(target) # [B, 1, 512, 512]
        timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (target.shape[0],)).long().to(device) # [B]
        noisy_target_images = noise_scheduler.add_noise(target, noise, timesteps).to(device) # [B, 1, 512, 512]
        model_input = torch.cat((noisy_target_images, condition), dim=1) #[B, 1+window_size, 512, 512]
        pred = model(model_input, timesteps).sample # [B, 1, 512, 512]
        loss = loss_fn(pred, noise)  # [B, 1, 512, 512] - [1, 1, 512, 512]
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
    torch.save(model.state_dict(), f"weights/diffusion_only_ddpm{epoch}.pth")

    avg_loss = sum(losses[-100:]) / 100
    print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")


@torch.no_grad()# it takes batched condition images
def generate_image(condition_images, model, noise_scheduler, device, num_inference_steps=50, guidance_scale=0.0):
    model.eval()
    if condition_images.ndim == 3:
        condition_images = condition_images.unsqueeze(0)
    condition_images = condition_images.to(device)

    target_shape = (condition_images.shape[0], 1, 512, 512)
    generated_images = torch.randn(target_shape, device=device)

    noise_scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(noise_scheduler.timesteps, desc="Inference"):
        model_input = torch.cat((generated_images, condition_images), dim=1)
        noise_pred = model(model_input, t).sample

        generated_images = noise_scheduler.step(noise_pred, t, generated_images).prev_sample

    # generated_images = torch.clamp(generated_images, -1.0, 1.0)

    model.train()
    return generated_images

for test_data in test_loader: # Doing batch prediction
    condition_img_tensors = test_data["input"].to(device)
    true_target_img_tensors = test_data["target"].to(device)

    generated_outputs_batch = generate_image(condition_img_tensors, model, noise_scheduler, device, num_inference_steps=1000)
    
    for i in range(generated_outputs_batch.shape[0]):
        single_condition_seq = condition_img_tensors[i].cpu() # [window_size, H, W]
        single_generated_output = generated_outputs_batch[i].cpu() # [1, H, W]
        single_true_target = true_target_img_tensors[i].cpu() # [1, H, W]

        plot_data = {
            'input': single_true_target,       
            'target': single_generated_output,   
            'lat': test_data['lat'][i].cpu(),  
            'lon': test_data['lon'][i].cpu()
        }
        plot_sample(plot_data, vmin=0, vmax=70, label1 = "truth", label2 = "generated", name = f"sample_ddpm_blues{i}")
        
    break