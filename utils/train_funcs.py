from torch import nn
import torch
import wandb
from tqdm.auto import tqdm
from dataset import DepictionDataset
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam, SGD
from datetime import datetime
import models
import gc



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
  

def build_model(model_name, dropout_p=0.7, pretrained=False):
    if model_name == "vgg16":
        model = models.build_model_vgg16(dropout_p=dropout_p, model_path=None)
    elif model_name == "vgg19":
        model = models.build_model_vgg19(dropout_p=dropout_p, model_path=None)
    elif model_name == "ResNet":
        model = models.ResNet(models.ResidualBlock, [3,4,6,3])
    model.apply(models.model_utils.init_weights)
    return model


def build_optimizer(model, initial_learning_rate):
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate, weight_decay = 0.001, momentum = 0.9)  
    return optimizer

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
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()
    tacc = tcorr/ttotal
    # print("[INFO] returning loss...")
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