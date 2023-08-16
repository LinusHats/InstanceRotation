from torch import nn
import torch
import wandb
from tqdm.auto import tqdm
from dataset import DepictionDataset
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from datetime import datetime



def build_dataloaders(base_path, batch_size):
    whole_data = DepictionDataset(annotations_file=f"{base_path}/labels.csv",
                              img_dir=f"{base_path}/TrainingInstanceCatalog", 
                              img_size=(224, 224))

    whole_data.set_mean((0.7826,)) 
    whole_data.set_std((0.2941,))

    train_split = 0.75
    test_split = 1-train_split
    try:
        train_data, test_data = random_split(dataset=whole_data,
                                            lengths=[int(len(whole_data)*train_split), int(len(whole_data)*test_split)+1],
                                            generator=torch.Generator().manual_seed(442))
    except:
        train_data, test_data = random_split(dataset=whole_data,
                                            lengths=[int(len(whole_data)*train_split), int(len(whole_data)*test_split)],
                                            generator=torch.Generator().manual_seed(442))
    finally:
        train_data, test_data = random_split(dataset=whole_data,
                                            lengths=[int(len(whole_data)*train_split)+1, int(len(whole_data)*test_split)],
                                            generator=torch.Generator().manual_seed(442))
    # generating the train and val splits
    # print("[INFO] generating train/val splits...")
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
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # print("[INFO] dataloaders created...")
    return train_dataloader, val_dataloader, test_dataloader
  

def build_optimizer(model, initial_learning_rate):
    optimizer = Adam(model.parameters(), lr=initial_learning_rate)
    return optimizer
    


def build_model_vgg16(dropout_p, model_path):
    # print("[INFO] loading model...")
    vgg16 = nn.Sequential(
                    nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),    # 0
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
                    nn.Dropout(dropout_p, False),                            # 41
                    nn.Linear(4096, 4096, True),                               # 42
                    nn.BatchNorm1d(4096),                                   # 43
                    nn.ReLU(True),                                          # 44
                    nn.Dropout(dropout_p, False),                            # 45
                    nn.Linear(4096, 8, True)                                 # 46
                )

    

    # print("[INFO] loading pretrained model...")
    pretrained_model = torch.load(model_path)
    # print("[INFO] initializing weights...")
    for i, j in zip([0, 2, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32], 
                [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]):
        vgg16[i].weight.data = pretrained_model.get_parameter(f'features.{j}.weight')
        vgg16[i].bias.data   = pretrained_model.get_parameter(f'features.{j}.bias')
    for i, j in zip([38, 42], 
                [0 ,  3]):
        vgg16[i].weight.data = pretrained_model.get_parameter(f'classifier.{j}.weight')
        vgg16[i].bias.data   = pretrained_model.get_parameter(f'classifier.{j}.bias')
        
    # print("\n[INFO] model build\n")
    
    return vgg16

def build_model_vgg19(dropout_p, model_path):
    vgg19 = nn.Sequential(
        nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),               # 0
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
        nn.Dropout(dropout_p, False),                 # 47
        nn.Linear(4096, 4096, True),                            # 48
        nn.BatchNorm1d(4096),                                   # 49
        nn.ReLU(True),                                          # 50
        nn.Dropout(dropout_p, False),                 # 51
        nn.Linear(4096, 8, True)            # 52
    )
    # print("[INFO] getting pretrained_model")
    pretrained_model = torch.load(model_path, map_location="cpu")
    # print("[INFO] setting weights and biases")
    for i, j in zip([0, 2, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 32, 34, 36, 38], 
            [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]):
        vgg19[i].weight.data = pretrained_model.get_parameter(f'features.{j}.weight')
        vgg19[i].bias.data   = pretrained_model.get_parameter(f'features.{j}.bias')
    for i, j in zip([44, 48],
                    [0 ,  3]):
        vgg19[i].weight.data = pretrained_model.get_parameter(f'classifier.{j}.weight')
        vgg19[i].bias.data   = pretrained_model.get_parameter(f'classifier.{j}.bias')
    # print("[INFO] Model build!")
    return vgg19

def train(model, train_loader, lossFunc, optimizer, epochs, device, val_dataloader, save=False):
    # Tell wnadb to watch  ehat the model gets up to
    wandb.watch(model, lossFunc, log="all", log_freq=10)
    
    
    # run training
    total_batches = len(train_loader) * epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    # print("[INFO] starting epoch...")
    last_val_loss = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_batch(train_loader, model, optimizer, lossFunc, device)
        example_ct += len(train_loader)
        val_loss, val_acc = validate_epoch(model, val_dataloader, lossFunc, device)
        train_log(epoch, train_loss, train_acc, val_loss, val_acc,  example_ct)
    
def validate_epoch(model, val_dataloader, lossFunc, device):
    vcorrect = vtotal = running_vloss = 0
    with torch.no_grad():
        for i, (vimages, vlabels) in enumerate(val_dataloader):
            vimages, vlabels = vimages.to(device), vlabels.to(device)
            voutputs = model(vimages)
            vloss = lossFunc(voutputs, vlabels)
            vloss = vloss.to("cpu")
            running_vloss += vloss
            _, predicted = torch.max(voutputs.data, 1)
            vtotal += vlabels.size(0)
            vcorrect += (predicted == vlabels).sum().item()
        avg_loss = running_vloss / (i+1)
        val_acc = vcorrect/vtotal
    return avg_loss, val_acc
       

def train_batch(train_loader, model, optimizer, lossFunc, device):
    ttotal = tcorr = 0
    for _, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)
        # print("[INFO] forward pass...")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        ttotal += labels.size(0)
        tcorr += (predicted == labels).sum().item()
        loss = lossFunc(outputs, labels)
        # print("[INFO] backward pass...")
        optimizer.zero_grad()
        loss.backward()
        # print("[INFO] optimizing...")
        optimizer.step()
        # print("[INFO] returning loss...")
    tacc = tcorr/ttotal
    return loss, tacc
    
def train_log(epoch,train_loss,train_acc, val_loss, val_acc ,example_ct):
    wandb.log({"epoch": epoch, 
              "train_loss": train_loss,
              "train_acc": train_acc,
              "val_loss": val_loss,
              "val_acc": val_acc
              }, step=example_ct)
    print(f"\tTRAIN:\t\t{train_loss:.3f}, {train_acc*100:.3f} %\n\tVALIDATION:\t{val_loss:.3f}, {val_acc*100:.3f} %")
    
    
def test(model, test_loader, device, base_path):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})
    # save the model locally in case...
    timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
  
    try:
        torch.save(model, f"{base_path}/models/{timestamp}.pth")   
    except:
        print("something went wrong saving the pth")
    try:
        torch.onnx.export(model, images, f"{base_path}/models/vgg16_InstanceCatalog.onnx")
        wandb.save(f"{base_path}/models/{timestamp}.onnx")
    except:
       print("[WARNING]: something went wrong trying to save the onnx file")