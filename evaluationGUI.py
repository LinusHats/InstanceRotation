import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
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
# model = torch.load(model_path, map_location=torch.device('cpu'))

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


# model = model.to(device)
# model.eval()

# y_true = []
# y_predictions = []
# with torch.no_grad():
#     correct = total = 0
#     for images, labels in tqdm(test_dataloader):
#         images, labels = images.to(device), labels.to(device)       
#         output = model(images)
#         # _, prediction = torch.max(output.data, 1)
#         y_true.extend(labels.tolist())
#         y_predictions.extend(output.tolist())
# y_pred = [i.index(max(i)) for i in y_predictions]
# wrong = [i for i, _ in enumerate(y_pred) if y_pred[i] != y_true[i]]

class ImageDisplayApp:
    def __init__(self, root, images, predicted_labels):
        self.root = root
        self.images = images
        self.current_index = 0
        self.pred = predicted_labels

        self.load_next_image()

        self.correct_button = ttk.Button(root, text="Correct", command=self.load_next_image)
        self.correct_button.pack(side="left", padx=10, pady=10)

        self.wrong_button = ttk.Button(root, text="Wrong", command=self.load_next_image)
        self.wrong_button.pack(side="right", padx=10, pady=10)

    def load_next_image(self):
        if self.current_index < len(self.images):
            original_image = self.images[self.current_index][0,:,:]
            prediction = self.pred[self.current_index]
            
            plt.imshow(original_image[0,:,:])
            plt.show()
            
            
            # transformed_image = self.transform_image(original_image[0,:,:], self.pred[self.current_index])
            transformed_image = original_image
            original_image = Image.fromarray(original_image)
            transformed_image = Image.fromarray(transformed_image)
            original_image = original_image.resize((200, 200))
            transformed_image = transformed_image.resize((200, 200))

            self.original_photo = ImageTk.PhotoImage(original_image)
            self.transformed_photo = ImageTk.PhotoImage(transformed_image)

            if hasattr(self, "original_label"):
                self.original_label.destroy()
                self.transformed_label.destroy()

            self.original_label = ttk.Label(self.root, image=self.original_photo)
            self.original_label.pack(side="left", padx=10, pady=10)

            self.transformed_label = ttk.Label(self.root, image=self.transformed_photo)
            self.transformed_label.pack(side="right", padx=10, pady=10)

            self.current_index += 1

            if self.current_index == len(self.images):
                self.correct_button.config(state="disabled")
                self.wrong_button.config(state="disabled")
                
            
    def transform_image(orig_image, prediction):
        flip = False if prediction < 4 else True
        if prediction == 0 or prediction == 4:
            angle = False
            angle_string = "0°"
        elif prediction == 1 or prediction == 5:
            angle = cv2.ROTATE_90_CLOCKWISE
            angle_string = "90°"
        elif prediction == 2 or prediction == 6:
            angle = cv2.ROTATE_180
            angle_string = "180°"
        elif prediction == 3 or prediction == 7:
            angle = cv2.ROTATE_90_COUNTERCLOCKWISE
            angle_string = "270°"
        transformed_image = cv2.rotate(orig_image, angle) if angle else orig_image
        transformed_image = cv2.flip(transformed_image, 1) if flip else transformed_image
        return transformed_image

# List of images which where predocited wrongly
wrong = np.load("wrong.npy")
y_correct = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")
y_predictions = np.load("y_predictions.npy")

images = [test_data[i][0].numpy() for i in wrong]
y_correct = [y_correct[i] for i in wrong]
y_pred = [y_pred[i] for i in wrong]
y_predictions = [y_predictions[i] for i in wrong]

root = tk.Tk()
root.title("Eval_Gui")
# plt.imshow(ImageDisplayApp.transform_image(images[0], y_pred[0]))
# plt.show()
app = ImageDisplayApp(root, images=images, predicted_labels=y_pred)

# root.mainloop()