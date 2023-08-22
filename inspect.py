from utils.train_funcs import build_dataloaders
import matplotlib.pyplot as plt
import numpy as np


base_path = r'C:\Users\linus\Documents\Datasets\InstanceRotation'

dataloader = build_dataloaders(base_path, 4)[0]

iterator = iter(dataloader)
img = next(iterator)[0]
one_channel = False
if one_channel:
    img = img.mean(dim=0)
img = img / 2 + 0.5  # unnormalize
npimg = img.numpy()
if one_channel:
    plt.imshow(npimg, cmap="Greys")
else:
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

plt.show()
