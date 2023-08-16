import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from utils import *

# log into wandb
import wandb


def main(config=None):

    # make behaviour deterministic
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
    # create sweep config

    
    # set device
    try:
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())
    except:
       if torch.backends.mps.is_available():
           device = torch.device("mps")
       else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device: ", device)
    base_path = r'C:\Users\lhartz\datasets\InstanceRotation\InstanceCatalogue\InstanceCatalog'

    
    with wandb.init(config=config):
        config = wandb.config
        

        model, train_dataloader, val_dataloader, test_loader, criterion, optimizer = make(config, base_path)
        model = model.to(device)
        criterion = criterion.to(device)
        # train the model
        # print("\nStarting to train model")
        train(model, train_dataloader, criterion, optimizer, config.epochs, device, val_dataloader)
    
        test(model, test_loader, device, base_path)
        
def make(config, base_path):

    # get the dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(base_path, config.batch_size)
    # print("\nStarting to build model")
    model = build_model_vgg19(config.dropout_p, f"{base_path}/vgg19_pretrained.pth")
    # print("\nStarting to build optimizer")
    optimizer = build_optimizer(model, config.initial_learning_rate)
    # print("\nStarting to build criterion")
    lossFunction = nn.CrossEntropyLoss()
    criterion = lossFunction
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer
  

if __name__ == "__main__":
    wandb.login(key="c3a6ab2da3aa3b00ec85e71846c96e5385899ac1")
    sweep_config = {
        'method': 'random',
        'name': 'GPU2_1',
        'project': 'VGG19_InstanceRotation',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'batch_size': {
                'values': [8, 12 ,16 , 25 , 32, 45 , 64]
            },
            'initial_learning_rate': {
                'distribution': 'uniform',
                'min': 0.000001,
                'max': 0.01
            },
            'epochs': {
                'value': 50,
            },
            'dropout_p': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.95,
            }
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
    wandb.agent(sweep_id, function=main, count=20)
    