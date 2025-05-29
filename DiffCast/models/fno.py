from neuralop.models import FNO
import torch.nn as nn
import torch

class FNOModel:
    def __init__(self, in_channels = 10, out_channels = 10):
        self.model = FNO(n_modes=(32, 32), hidden_channels=64,
                in_channels=in_channels, out_channels=out_channels).to("cuda" if torch.cuda.is_available() else "cpu")
    def predict(self, frames_in, frames_gt = None, compute_loss = False):
        if frames_in.shape[2] == 1:
            frames_in = frames_in.squeeze(2)
        # print(f"Input shape: {frames_in.shape}")
        out = self.model(frames_in)
        out = out.unsqueeze(2)
        # print(f"Output shape: {out.shape}")
        loss = None
        if compute_loss and frames_gt is not None:
            loss = nn.MSELoss()(out, frames_gt)
        return out, loss
    
if __name__ == "__main__":
    model = FNOModel()
    inp = torch.randn(8, 10, 1, 384, 384)
    gt = torch.randn(8, 10, 1, 384, 384)
    out, loss = model.predict(inp, gt, compute_loss=True)
    print(out.shape, loss.item())