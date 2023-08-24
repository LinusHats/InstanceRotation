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
    base_path = r'C:\Users\lhartz\datasets\InstanceRotation'

    wandb.init(    
        project="Instance",
        name="PaddedInstances_100Epochs",
        config = {
        "epochs": 100,
        "initial_learning_rate": 0.046,
        "dropout_p": 0.2,
        "batch_size": 128,
    })
    
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
    model = build_model(model_name="ResNet", dropout_p=config.dropout_p)
    # print("\nStarting to build optimizer")
    optimizer = build_optimizer(model, config.initial_learning_rate)
    # print("\nStarting to build criterion")
    lossFunction = nn.CrossEntropyLoss()
    criterion = lossFunction
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer
  

if __name__ == "__main__":
    wandb.login(key="c3a6ab2da3aa3b00ec85e71846c96e5385899ac1")
    main()
    