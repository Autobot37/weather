import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config_path = None):
        super().__init__()

    def prepare_data(self):
        # (runs on single GPU/CPU)
        pass

    def setup(self, stage=None):
        #  every GPU/process
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def plot_batch(self, split='train', batch_idx = 0):
        pass


if __name__ == "__main__":
    dm = BaseDataModule()
    dm.prepare_data()
    dm.setup()
    
