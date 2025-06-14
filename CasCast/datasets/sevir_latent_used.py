import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
try:
    from petrel_client.client import Client
except:
    pass
import io


def get_sevir_latent_dataset( split, input_length=13, pred_length=12, data_dir='/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir', base_freq='5min', height=384, width=384, **kwargs):
    return sevir_latent(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)



class sevir_latent(Dataset):
    def __init__(self, split, input_length=13, pred_length=12,  base_freq='5min', height=384, width=384,
                 latent_size='48x48x4', 
                 data_dir='None', latent_gt_dir='None', latent_deterministic_dir='None',
                 **kwargs):
        super().__init__()
        assert input_length == 13, pred_length==12
        self.input_length = 13
        self.pred_length = 12

        self.latent_size = latent_size

        self.file_list = self._init_file_list(split)
        self.data_dir = os.path.join(data_dir, f'{split}_2h')
        self.latent_gt_dir = os.path.join(latent_gt_dir, f'{split}_2h')
        self.latent_deterministic_dir = os.path.join(latent_deterministic_dir, f'{split}_2h')

    def _init_file_list(self, split):
        if split == 'train':
            txt_path = '/home/vatsal/Dataserver/cascast/output/train.txt'
        elif split == 'valid':
            txt_path = '/home/vatsal/Dataserver/cascast/output/val.txt'
        elif split == 'test':
            txt_path = '/home/vatsal/Dataserver/cascast/output/test.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        return files
    
    def __len__(self):
        return len(self.file_list)

    def _load_latent_frames(self, file, datasource):
        file_path = os.path.join(datasource, file)
        frame_data = np.load(file_path)
        ## t, c, h, w ##
        tensor = torch.from_numpy(frame_data)
        return tensor

    def _load_frames(self, file):
        file_path = os.path.join(self.data_dir, file)
        frame_data = np.load(file_path)
        tensor = torch.from_numpy(frame_data) / 255
        # print(colored(f"min: {tensor.min().item()}, max: {tensor.max().item()}", 'green'))
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.unsqueeze(3)
        tensor = tensor.permute(2, 3, 0, 1)
        return tensor
    
    def __getitem__(self, index):
        file = self.file_list[index]
        gt_data = self._load_frames(file)[self.input_length:]
        gt_latent_data = self._load_latent_frames(file, datasource=self.latent_gt_dir)
        coarse_latent_data = self._load_latent_frames(file, datasource=self.latent_deterministic_dir)
        packed_results = dict()
        packed_results['inputs'] = coarse_latent_data
        packed_results['data_samples'] = {'latent':gt_latent_data, 'original':gt_data}
        return packed_results