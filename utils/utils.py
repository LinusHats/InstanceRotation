import torchvision.models as models
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def get_pretrained(base_path, model="vgg19"):
    vgg19 = models.vgg19(pretrained=True)
    torch.save(vgg19, f"{base_path}/vgg19_pretrained.pth")


def get_mean_std(dataset):
    mean = 0.
    std = 0.
    dataloader = DataLoader(dataset)
    for image,_ in tqdm(dataloader):
        # image = dataset[i][0]
        mean += torch.mean(image)
        std += torch.std(image)
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
