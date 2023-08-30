import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataset.dataset import DepictionDataset
import onnx
from  sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import pandas as pd
from torch.utils.data import random_split, DataLoader

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# resnet tested on padded dataset
model_path        = r"C:\Users\linus\Documents\models\instanceRotation\ResNet34_57perc_20230824_16-49-09.pth"
dataset_base_path = r"C:\Users\linus\Documents\Datasets\InstanceRotation"
batch_size=128
# Load the ONNX model
# model = onnx.load(model_path)

# Check that the model is well formed
# onnx.checker.check_model(model)

# Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

# load the model
model = torch.load(model_path, map_location=torch.device('cpu'))

whole_data = DepictionDataset(annotations_file=f"{dataset_base_path}/PaddedTraining.csv",
                            img_dir=f"{dataset_base_path}/PaddedTraining", 
                            img_size=(224, 224))

whole_data.set_mean((0.7826,)) 
whole_data.set_std((0.2941,))

train_split = 0.75
test_split = 1-train_split


# make the test data from the dataset
train_data, test_data = random_split(dataset=whole_data,
                                     lengths=[int(len(whole_data)*train_split), int(len(whole_data)*test_split)+1],
                                     generator=torch.Generator().manual_seed(442))

test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)


model = model.to(device)
model.eval()
print(model)

y_true = []
y_pred = []
with torch.no_grad():
    correct = total = 0
    for images, labels in tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device)       
        output = model(images)
        # _, prediction = torch.max(output.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(output.tolist())
        # total += label.size(0)
        # correct += (prediction == label).sum().item()
        
y_pred = [i.index(max(i)) for i in y_pred]
print(classification_report(y_true, y_pred))
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

plt.imshow(conf_matrix/conf_matrix.max(), cmap="gist_ncar")
# change the xticks to be the class names, not the indices
plt.xticks(range(8), list(DepictionDataset.labels_map.values()), rotation=90)
plt.yticks(range(8), list(DepictionDataset.labels_map.values()))
# add the confusion scores to the plot
for i in range(8):
    for j in range(8):
        plt.text(j, i, f'{conf_matrix[i, j]/conf_matrix.max()*100:.2f}', ha="center", va="center", color="white")
plt.xlabel("Predicted")

        

