from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import torch.nn as nn

class PWModel(nn.Module):
    def __init__(self, image_size: int = 512, condition_window_size: int = 4, prediction_window_size: int = 4):
        super().__init__()
        self.image_size = image_size
        self.condition_window_size = condition_window_size
        self.prediction_window_size = prediction_window_size
        # in_channels: `prediction_window_size` (for noisy targets) + `condition_window_size` (for condition frames)
        # out_channels: `prediction_window_size` (model predicts noise for each target frame)
        self.unet = UNet2DModel(
            sample_size=self.image_size,
            in_channels=self.prediction_window_size + self.condition_window_size,
            out_channels=self.prediction_window_size,
            layers_per_block=2,
            block_out_channels=(32, 32, 64, 64, 128, 128),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, model_input: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(model_input, timestep).sample