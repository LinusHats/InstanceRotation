import torchvision.models as models
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_pretrained(base_path, model="vgg19"):
    if model == "vgg19":    
        vgg19 = models.vgg19(pretrained=True)
        torch.save(vgg19, f"{base_path}/vgg19_pretrained.pth")
    elif model == "vgg16":
        vgg16 = models.vgg16(pretrained=True)
        torch.save(vgg16, f"{base_path}/vgg16_pretrained.pth")


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

def print_vgg16():
    vgg16 = models.vgg16()
    print(vgg16)

def print_vgg19():
    vgg19 = models.vgg19()
    print(vgg19)
        
