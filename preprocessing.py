
from dataset import DepictionDataset
from utils import get_mean_std, get_pretrained, print_vgg16
from models import ResNet50


# base_path = r'C:\Users\lhartz\datasets\InstanceRotation\InstanceCatalogue\InstanceCatalog'


# print(get_mean_std(DepictionDataset(annotations_file=f"{base_path}/labels.csv",
#                               img_dir=f"{base_path}/TrainingInstanceCatalog", 
#                               img_size=(224, 224))
# ))

# get_pretrained(base_path, model="vgg16")

# print_vgg16()


# resnet18 = ResNet50.ResNet(3, ResNet50.ResBlock, [2,2,2,2], useBottleneck=False, outputs=8)
