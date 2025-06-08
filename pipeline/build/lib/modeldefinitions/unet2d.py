from diffusers import UNet2DModel

def get_unet2d(image_size=64, in_ch=2, out_window=1, num_train_timesteps=1000):
    model = UNet2DModel(
            sample_size=image_size,
            in_channels=in_ch,
            out_channels=out_window,
            center_input_sample=False,
            time_embedding_type="positional",
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            mid_block_type="UNetMidBlock2D",
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(32, 32, 64),
            layers_per_block=1,
            mid_block_scale_factor=1,
            downsample_padding=1,
            downsample_type="conv",
            upsample_type="conv",
            dropout=0.0,
            act_fn="silu",
            attention_head_dim=4,
            norm_num_groups=16,
            norm_eps=1e-5,
            resnet_time_scale_shift="default",
            num_train_timesteps=num_train_timesteps,
        )
    return model