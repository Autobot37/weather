import torch
from collections import OrderedDict
from termcolor import colored

def load_checkpoint_cascast(checkpoint_path, model):
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint_dict['model']
    ckpt_submodels = list(checkpoint_model.keys())
    print(ckpt_submodels)
    submodels = ['autoencoder_kl']
    key = 'autoencoder_kl'
    if key not in submodels:
        print(f"warning!!!!!!!!!!!!!: skip load of {key}")
    new_state_dict = OrderedDict()
    for k, v in checkpoint_model[key].items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("net."):
            name = name[len("net."):]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    print(colored(f"loaded {key} successfully the game is on", 'green'))
    return model