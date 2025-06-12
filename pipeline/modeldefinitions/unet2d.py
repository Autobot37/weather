from diffusers import UNet2DModel
import torch
import torch.nn as nn

class WrappedUNet2D(nn.Module):
    """
    Simplified wrapper for diffusers.UNet2DModel:
      - Main init params: sample_size, in_channels, out_channels, num_train_timesteps
      - All other architectural choices defaulted for a robust U-Net with attention
    """
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        num_train_timesteps: int,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_train_timesteps = num_train_timesteps

        # Default U-Net configuration
        default_channels = (64, 128, 128, 256)
        default_down = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        default_up =("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels * 2,
            out_channels=out_channels,
            center_input_sample=False,
            time_embedding_type="positional",
            down_block_types=default_down,
            mid_block_type="UNetMidBlock2D",
            up_block_types=default_up,
            block_out_channels=default_channels,
            layers_per_block=2,
            downsample_type="conv",
            upsample_type="conv",
            act_fn="silu",
            norm_num_groups=32,
            attention_head_dim=8,
            dropout=0.1,
            resnet_time_scale_shift="default",
            num_train_timesteps=num_train_timesteps,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x    -- noisy input [B, C, H, W]
            t    -- timestep [B] or int
            cond -- optional condition [B, C, H, W]
        Returns:
            predicted noise of shape [B, out_channels, H, W]
        """
        if cond is None:
            cond = torch.zeros_like(x)
        inp = torch.cat((cond, x), dim=1)
        return self.unet(inp, timestep=t).sample

model = WrappedUNet2D(sample_size=128, in_channels=10, out_channels=10, num_train_timesteps=1000)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
for i in range(10):
    x = torch.randn(4, 10, 128, 128).to('cuda')  # [B, C, H, W]
    t = torch.randint(0, 1000, (4,)).to('cuda')  # Random timesteps
    cond = torch.randn(4, 10, 128, 128).to('cuda')  # Optional condition
    out = model(x, t, cond)
    print(f"Output shape: {out.shape}")  # Should be [B, out_channels, H, W]