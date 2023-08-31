import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class DepictionDataset(Dataset):
    labels_map = {
        0: "Not Flipped, 0°",
        1: "Not Flipped, 90°",
        2: "Not Flipped, 180°",
        3: "Not Flipped, 270°",
        4: "Flipped, 0°",
        5: "Flipped, 90°",
        6: "Flipped, 180°",
        7: "Flipped, 270°"
    }

    def __init__(self, annotations_file, img_dir, img_size):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.classes = ["Not Flipped, 0°", "Not Flipped, 90°", "Not Flipped, 180°",
                "Not Flipped, 270°", "Flipped, 0°", "Flipped, 90°",
                "Flipped, 180°", "Flipped, 270°"]
        self.mean = None
        self.std = None

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.mean and self.std:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
            ])
        else:

            transform = transforms.Compose([
            transforms.ToTensor(),Training finalinstance
            ])
        
        img_path = os.path.join(self.img_dir,
                                self.img_labels.iloc[idx, 0])
        # print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, self.img_size)
        img_tensor = transform(image)
        label = self.img_labels.iloc[idx, 1]
        return img_tensor, label
    
    def set_mean(self, mean):
        self.mean = mean
        
    def set_std(self, std):
        self.std = std

    
    
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