import torch
import torch.nn as nn
from diffusers import UNet3DConditionModel
"""
8 hrs per epoch on A6000 48 GB
"""
class CondEncoder(nn.Module):
    """
    Encodes a conditional tensor of shape (B, T2, C, H, W)
    into encoder hidden states (B, seq_len, feature_dim).
    """
    def __init__(self, in_channels=4, feature_dim=512):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, cond):
        # cond: (B, T2, C, H, W) -> (B, C, T2, H, W)
        x = cond.permute(0, 2, 1, 3, 4)
        x = self.conv_blocks(x)  # (B, feature_dim, T2, H, W)
        b, f, t2, h, w = x.shape
        seq = x.flatten(2).permute(0, 2, 1)  # (B, T2*H*W, feature_dim)
        return seq

class CustomUNet3D(nn.Module):
    """
    UNet3D with reduced memory footprint:
      - smaller block sizes
      - gradient checkpointing & forward‐chunking
    """
    def __init__(
        self,
        sample_size: int = 48,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_feature_dim: int = 512,
        cross_attention_dim: int = 512,
        attention_head_dim: int = 64,
    ):
        super().__init__()

        # 3D UNet backbone
        self.unet = UNet3DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=[
                "DownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ],
            up_block_types=[
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "UpBlock3D",
            ],
            block_out_channels=[64, 128, 256, 256],
            layers_per_block=1,
            mid_block_scale_factor=1.0,
            act_fn="silu",
            norm_num_groups=16,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )
        # conditional encoder
        self.cond_encoder = CondEncoder(
            in_channels=in_channels,
            feature_dim=encoder_feature_dim,
        )

    def forward(self, noisy, timesteps, cond, **kwargs):
        # noisy: (B, T, C, H, W) -> (B, C, T, H, W)
        noisy_in = noisy.permute(0, 2, 1, 3, 4)
        # encode cond -> (B, seq_len, feature_dim)
        encoder_hidden_states = self.cond_encoder(cond)
        return self.unet(
            sample=noisy_in,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

# Example usage
if __name__ == "__main__":
    B, Tn, Tc, H, W = 1, 13, 4, 48, 48
    noisy = torch.randn(B, Tn, Tc, H, W).cuda()
    cond  = torch.randn(B, 12, Tc, H, W).cuda()
    timesteps = torch.tensor([10] * B).cuda()
    model = CustomUNet3D().cuda()
    out = model(noisy, timesteps, cond)
    print(out.sample.shape)  # (B, out_channels, Tn, H, W)
