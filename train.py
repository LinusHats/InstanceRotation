from tqdm import tqdm
import warnings
from skimage import transform
from sklearn.metrics import classification_report
from torchvision.utils import make_grid
import torchvision.models as models
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import torch
import torch.nn.functional as F
# import torch_directml
from datetime import datetime
import time
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from data import DepictionDataset
import data
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import yaml
warnings.filterwarnings("ignore") # __/OO\__

with open("./config.yml", "r") as f:
    config = yaml.safe_load(f)
assert config is not None, "Config file not found!"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

summary_path = f"{config['BASE_PATH']}/logs/LOG_{config['SUMMARY_NAME']}_{timestamp}.txt"
model_path = f"{config['BASE_PATH']}/models/MODEL_{config['SUMMARY_NAME']}_{timestamp}.pth"
writer_path =f"{config['BASE_PATH']}/runs/{config['SUMMARY_NAME']}_{timestamp}"

if config["RESUME"]:
    summary_path = f"{config['BASE_PATH']}/logs/LOG_RESUME_{config['SUMMARY_NAME']}_{timestamp}.txt"
    model_path = f"{config['BASE_PATH']}/models/MODEL_RESUME_{config['SUMMARY_NAME']}_{timestamp}.pth"
    writer_path =f"{config['BASE_PATH']}/runs/RESUME_{config['SUMMARY_NAME']}_{timestamp}"
    
    with open(f"{config['BASE_PATH']}/logs/LOG_{config['RESUME_SUMMARY_NAME']}.txt", "r") as f:
        lines = f.readlines()
    f.close()
    
    with open(f"{config['BASE_PATH']}/logs/LOG_RESUME_{config['RESUME_SUMMARY_NAME']}.txt", "w+") as f:
        f.write("# RESUMING TRAINING\n")
        f.writelines(lines)
        f.flush()

### Set the Device for training ###
if config["MACHINE"] == "NPU":
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] ``device`` was set to {}".format(device))
### Get the models ready ###


pretrained_models = {
    "vgg11" : torch.load(f"{config['BASE_PATH']}/pretrained_models/vgg11_pretrained.pth"),
    "vgg16" : torch.load(f"{config['BASE_PATH']}/pretrained_models/vgg16_pretrained.pth"),
    "vgg19" : torch.load(f"{config['BASE_PATH']}/pretrained_models/vgg19_pretrained.pth")
}

vgg11 = nn.Sequential(
    nn.Conv2d(config["NUM_CHANNELS"], 64, (3, 3), (1, 1), (1, 1)),    # 0
    nn.BatchNorm2d(64),                                     # 1
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 2
    nn.ReLU(True),                                          # 3
    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),                 # 4
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 5
    nn.ReLU(True),                                          # 6
    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),                # 7
    nn.ReLU(True),                                          # 8
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),                # 9
    nn.BatchNorm2d(256),                                    # 10
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 11
    nn.ReLU(True),                                          # 12
    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),                # 13
    nn.ReLU(True),                                          # 14
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),                # 15
    nn.BatchNorm2d(512),                                    # 16
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 17
    nn.ReLU(True),                                          # 18
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),                # 19
    nn.ReLU(True),                                          # 20
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),                # 21
    nn.BatchNorm2d(512),                                    # 22
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 23
    nn.ReLU(True),                                          # 24
    nn.AdaptiveAvgPool2d((7, 7)),                           # 25
    nn.Flatten(1, -1),                                      # 26
    nn.Linear(512 * 7 * 7, 4096, True),                         # 27
    nn.BatchNorm1d(4096),                                   # 28
    nn.ReLU(True),                                          # 29
    nn.Dropout(config["DROPOUT_P"], False),                 # 30
    nn.Linear(4096, 4096, True),                                # 31
    nn.BatchNorm1d(4096),                                   # 32
    nn.ReLU(True),                                          # 33
    nn.Dropout(config["DROPOUT_P"], False),                 # 34
    nn.Linear(4096, config["NUM_CLASSES"], True)            # 35
)

vgg16 = nn.Sequential(
    nn.Conv2d(config["NUM_CHANNELS"], 64, (3, 3), (1, 1), (1, 1)),    # 0
    nn.ReLU(True),                                          # 1
    nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),              # 2
    nn.BatchNorm2d(64),                                     # 3
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 4
    nn.ReLU(True),                                          # 5
    
    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),             # 6
    nn.ReLU(True),                                          # 7
    nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),            # 8
    nn.BatchNorm2d(128),                                    # 9
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 10
    nn.ReLU(True),                                          # 11
    
    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),            # 12
    nn.ReLU(True),                                          # 13
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 14
    nn.ReLU(True),                                          # 15
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 16
    nn.BatchNorm2d(256),                                    # 17
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 18
    nn.ReLU(True),                                          # 19
    
    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),            # 20
    nn.ReLU(True),                                          # 21
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 22
    nn.ReLU(True),                                          # 23        
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 24
    nn.BatchNorm2d(512),                                    # 25
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 26
    nn.ReLU(True),                                          # 27
    
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 28
    nn.ReLU(True),                                          # 29
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 30
    nn.ReLU(True),                                          # 31
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 32
    nn.BatchNorm2d(512),                                    # 33
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 34
    nn.ReLU(True),                                          # 35
    
    nn.AdaptiveAvgPool2d((7, 7)),                           # 36
    nn.Flatten(1, -1),                                      # 37
    nn.Linear(512 * 7 * 7, 4096, True),                     # 38
    nn.BatchNorm1d(4096),                                   # 39
    nn.ReLU(True),                                          # 40
    nn.Dropout(config["DROPOUT_P"], False),                 # 41
    nn.Linear(4096, 4096, True),                            # 42
    nn.BatchNorm1d(4096),                                   # 43
    nn.ReLU(True),                                          # 44
    nn.Dropout(config["DROPOUT_P"], False),                 # 45
    nn.Linear(4096, config["NUM_CLASSES"], True)            # 46
)

vgg19 = nn.Sequential(
    nn.Conv2d(config["NUM_CHANNELS"], 64, (3, 3), (1, 1), (1, 1)),    # 0
    nn.ReLU(True),                                          # 1
    nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),              # 2
    nn.BatchNorm2d(64),                                     # 3
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 4
    nn.ReLU(True),                                          # 5
    
    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),             # 6
    nn.ReLU(True),                                          # 7
    nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),            # 8
    nn.BatchNorm2d(128),                                    # 9
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 10
    nn.ReLU(True),                                          # 11
    
    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),            # 12
    nn.ReLU(True),                                          # 13
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 14
    nn.ReLU(True),                                          # 15
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 16
    nn.ReLU(True),                                          # 17
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),            # 18
    nn.BatchNorm2d(256),                                    # 19
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 20
    nn.ReLU(True),                                          # 21
    
    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),            # 22
    nn.ReLU(True),                                          # 23
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 24
    nn.ReLU(True),                                          # 25
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 26
    nn.ReLU(True),                                          # 27
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 28
    nn.BatchNorm2d(512),                                    # 29
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 30
    nn.ReLU(True),                                          # 31
    
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 32
    nn.ReLU(True),                                          # 33
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 34
    nn.ReLU(True),                                          # 35
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 36
    nn.ReLU(True),                                          # 37
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),            # 38
    nn.BatchNorm2d(512),                                    # 39
    nn.MaxPool2d(2, 2, 0, 1, False, False),                 # 40
    nn.ReLU(True),                                          # 41
    
    nn.AdaptiveAvgPool2d((7, 7)),                           # 42
    
    nn.Flatten(1, -1),                                      # 43
    nn.Linear(512 * 7 * 7, 4096, True),                     # 44
    nn.BatchNorm1d(4096),                                   # 45
    nn.ReLU(True),                                          # 46
    nn.Dropout(config["DROPOUT_P"], False),                 # 47
    nn.Linear(4096, 4096, True),                            # 48
    nn.BatchNorm1d(4096),                                   # 49
    nn.ReLU(True),                                          # 50
    nn.Dropout(config["DROPOUT_P"], False),                 # 51
    nn.Linear(4096, config["NUM_CLASSES"], True)            # 52
)

if config["RESUME"]:
    print("Resuming model from", config["RESUME_MODEL"])
    model = torch.load(config["RESUME_MODEL"])
else:
    def initialize_vgg11():
        for i, j in zip([0, 4, 7, 9, 13, 15, 19, 21], [0, 3, 6, 8, 11, 13, 16, 18]):
            vgg11[i].weight.data = (pretrained_models["vgg11"].get_parameter(f'features.{j}.weight'))
            vgg11[i].bias.data = (pretrained_models["vgg11"].get_parameter(f'features.{j}.bias'))
            
        for i,j in zip([27, 31], [0, 3]):
            vgg11[i].weight.data = (pretrained_models["vgg11"].get_parameter(f'classifier.{j}.weight'))
            vgg11[i].bias.data = (pretrained_models["vgg11"].get_parameter(f'classifier.{j}.bias'))

    def initialize_vgg16():
        for i, j in zip([0, 2, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32], 
                        [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]):
            vgg16[i].weight.data = pretrained_models["vgg16"].get_parameter(f'features.{j}.weight')
            vgg16[i].bias.data   = pretrained_models["vgg16"].get_parameter(f'features.{j}.bias')
        for i, j in zip([38, 42], 
                        [0 ,  3]):
            vgg16[i].weight.data = pretrained_models["vgg16"].get_parameter(f'classifier.{j}.weight')
            vgg16[i].bias.data   = pretrained_models["vgg16"].get_parameter(f'classifier.{j}.bias')
            
    def initialize_vgg19():
        for i, j in zip([0, 2, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 32, 34, 36, 38], 
                        [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]):
            vgg19[i].weight.data = pretrained_models["vgg19"].get_parameter(f'features.{j}.weight')
            vgg19[i].bias.data   = pretrained_models["vgg19"].get_parameter(f'features.{j}.bias')
        for i, j in zip([44, 48],
                        [0 ,  3]):
            vgg19[i].weight.data = pretrained_models["vgg19"].get_parameter(f'classifier.{j}.weight')
            vgg19[i].bias.data   = pretrained_models["vgg19"].get_parameter(f'classifier.{j}.bias')
        
    if config["MODEL"] == "VGG11":
        model = vgg11
        if config["PRETRAINED"]:
            print("[INFO] Initializing VGG11 with pretrained weights")
            initialize_vgg11()
    elif config["MODEL"] == "VGG16":
        model = vgg16
        if config["PRETRAINED"]:
            initialize_vgg16()
            print("[INFO] Initializing VGG16 with pretrained weights")
    elif config["MODEL"] == "VGG19":
        model = vgg19
        if config["PRETRAINED"]:
            initialize_vgg19()
            print("[INFO] Initializing VGG19 with pretrained weights")

model = model.to(device)

### Get the data ready  ###
classes = {
        0: "Not Flipped, 0°",
        1: "Not Flipped, 90°",
        2: "Not Flipped, 180°",
        3: "Not Flipped, 270°",
        4: "Flipped, 0°",
        5: "Flipped, 90°",
        6: "Flipped, 180°",
        7: "Flipped, 270°"
    }
whole_data = DepictionDataset(annotations_file=config["LABEL_FILE_PATH"],
                              img_dir=config["IMAGE_DIR_PATH"], 
                              img_size=(224, 224))
print(len(whole_data))

whole_data.set_mean((0.5,)) # TODO Check if that realy the case
whole_data.set_std((0.5,))

print("[INFO] making train and test data from the dataset...")
train_data, test_data = random_split(dataset=whole_data,
                                    lengths=[int(len(whole_data)*0.7)+1, int(len(whole_data)*0.3)],
                                    generator=torch.Generator().manual_seed(442))

# generating the train and val splits
print("[INFO] generating train/val splits...")
numTrainSamples = int(len(train_data) * 0.8)
numValSamples   = int(len(train_data) * 0.2  ) + 1
(training_data, validation_data) = random_split(dataset=train_data,
                                                lengths=[numTrainSamples, numValSamples],
                                                generator=torch.Generator().manual_seed(322))

print(f"[INFO]:" \
      f"\n\tthe train data has length {len(train_data)}" \
      f"\n\tthe val   data has length {len(validation_data)}"\
      f"\n\tthe test  data has length {len(test_data)}\n")


# initialize the train, validation and test data loaders
train_dataloader = DataLoader(training_data, batch_size=config['BATCH_SIZE'], shuffle=True)
val_dataloader   = DataLoader(validation_data, batch_size=config['BATCH_SIZE'], shuffle=False)
test_dataloader  = DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=True)
    
# initialize the optimizer
print("\n[INFO] initializing the optimizer and the loss function")
optimizer = Adam(model.parameters(), lr=config["INIT_LR"])
lossFunc = nn.CrossEntropyLoss() 
    
epoch_stats = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
iteration_stats = {"iteration": [], "train_loss": []}

writer = SummaryWriter(log_dir=writer_path)


### write own logfile ###
summary_comment = [f"COMMENT:       {config['SUMMARY_COMMENT']}",
                   f"TIMESTAMP:     {timestamp}"
                   f"MACHINE:       {config['MACHINE']}",
                   f"MODEL:         {config['MODEL']}", 
                   f"PRETRAINED:    {config['PRETRAINED']}",
                   f"INIT_LR:       {config['INIT_LR']}", 
                   f"BATCH_SIZE:    {config['BATCH_SIZE']}",
                   f"DROPOUT_P:     {config['DROPOUT_P']}",
                   ]             

with open(summary_path, "w+") as f:
    f.truncate(0)
    model_string = str(model).split('\n')
    for line in [*summary_comment, *model_string]:
        f.write(f"# {line} \n")
    f.write("epoch,train_loss,val_loss,accuracy\n")
    f.flush()
    
### Training Loop ###


best_vloss = 1_000_000.

epoch_number = 1
if config["RESUME"]:
    epoch_number = config["RESUME_EPOCH"]

for epoch in tqdm(range(config["EPOCHS"]), position=0, desc='Epochs'):
    # clear_output(wait=True)
    # make shure gradient tracking is on, and pass over the data
    model.train(True)
    
    start_time = time.time()
    running_loss = 0.
     
    # Train one batch
    for batch, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=1, desc='Batches', leave=True):
        iteration_number = epoch * len(train_dataloader) + batch
        # split the input into data and label
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
    
        # Zero the gradients for each batch
        optimizer.zero_grad()

        # Make predictions for the batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = lossFunc(outputs, labels)
        loss.backward()

        # Adjust the learning weights
        optimizer.step()


        # Collect data and report
        running_loss +=  loss.item()
        if batch%10 == 9: # doing it with 999 instead of 0 avoids the first batch to be counted
            # display.clear_output(wait=True)
            last_loss = running_loss / 10 # loss per batch
            writer.add_scalar('Train loss vs. iteration', last_loss, iteration_number)
            iteration_stats['iteration'].append(iteration_number)
            iteration_stats['train_loss'].append(last_loss)
            running_loss = 0.
    end_time = time.time()
    avg_loss = last_loss
    # Perform validation
    running_vloss = 0.
    # Set model to eval mode,  disabling dropout and using population statistics for batch normalization
    model.eval()
    # Disable gradient computation and reduce memory consumption.
    vcorrect = 0
    vtotal = 0
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device) 
            voutputs = model(vinputs)
            vloss = lossFunc(voutputs,  vlabels)
            vloss = vloss.to("cpu")
            running_vloss += vloss
            _, predicted = torch.max(voutputs.data, 1)
            vtotal += vlabels.size(0)
            vcorrect += (predicted == vlabels).sum().item()
    avg_vloss = running_vloss / (i+1)
    # print(f'LOSS train {avg_loss:.5f} valid {avg_vloss:.5f}')

    # Log the running loss averaged per batch for both training and validations
    epoch_stats['epoch'].append(epoch_number)
    epoch_stats['train_loss'].append(avg_loss) 
    epoch_stats['val_loss'].append(avg_vloss)
    epoch_stats['val_acc'].append(vcorrect/vtotal)

    writer.add_scalars('Losses per epoch', 
                        {'Training' : avg_loss, 'Validation' : avg_vloss},
                        epoch_number +1)
    writer.add_scalars('Accuracy per epoch',
                       {'Accuracy' : vcorrect/vtotal},
                       epoch_number +1)
    writer.flush()
    with open(summary_path, 'a') as f:
        f.write(f'{epoch_number},{avg_loss},{avg_vloss},{vcorrect/vtotal}\n')
    torch.save(model,  f"{config['BASE_PATH']}/models/LAST_{config['SUMMARY_NAME']}_{timestamp}.pth")
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"{config['BASE_PATH']}/models/BEST_{config['SUMMARY_NAME']}_{timestamp}.pth"
        torch.save(model,  model_path)
    
    epoch_number += 1

def test(data_loader=test_dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the {total} test images: {100*correct/total:.4f} %')
    return correct, total


correct, total = test()
with open(summary_path, "a") as f:
    f.write(
f"""
# |###################################################|
# |#################     Results      ################|
# |###################################################|
# 
# Total size of the dataset:      {len(whole_data)}
# Size of the training dataset:   {len(train_data)}
# Size of the validation dataset: {len(validation_data)}
# Size of the test dataset:       {len(test_data)}
# Test Accuracy:                  {100*correct/total:.4f} %""")