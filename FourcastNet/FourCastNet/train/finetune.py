import os
import sys
import time
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unlog_tp_torch, top_quantiles_error_torch
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet, PrecipNet
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import torch
from model.afnonet import AFNONet  # update based on your module structure
from my_lightning_module import LitAFNONet  # your LightningModule
from omegaconf import OmegaConf

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname, weights_only= False)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def train_model(model, params, train_loader, val_loader, optimizer, scheduler, start_epoch, end_epoch):
    # Initialize the model
    model.train()
    scaler = amp.GradScaler()
    for epoch in range(start_epoch, end_epoch):
        for i, data in enumerate(train_loader):
            inputs = data['input'].to(params.device)
            targets = data['target'].to(params.device)
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if i % params.log_interval == 0:
                logging.info(f'Epoch [{epoch}/{end_epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
        scheduler.step()


ckpt_path = '/home/vatsal/NWM/FourcastNet/FourCastNet/training_checkpoints/backbone.ckpt'
# model = AFNONet.load_from_checkpoint(ckpt_path) # or 'cuda' if using GPU
model = AFNONet.load_from_checkpoint(ckpt_path, layers=3, drop_rate=0)