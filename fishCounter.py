import argparse, subprocess, os, random, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

class CountingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict, transforms=None):
        self.data_list = list(data_dict.keys())
        self.data_dict = data_dict
        self.transforms = transforms

    def __getitem__(self, index):
       
        with open(self.data_list[i], 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transforms(img), self.data_dict[self.data_list[index]]

    def __len__(self):
        return len(self.data_dict)

def prepareData():

    # Define transforms
    data_transforms = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.9,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([100,100,100],[0.5,0.5,0.5])]),
    'val': transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([100,100,100],[0.5,0.5,0.5])])
    }

    image_data = {}
    image_data['val'] = {}
    image_data['train'] = {}

    for project in [x for x in os.listdir('CountingData/') if x[0] != '.']:
        for video in [x for x in os.listdir('CountingData/' + project) if x[0] != '.']:
            for label in [x for x in os.listdir('CountingData/' + project + '/' + video) if x[0] != '.']:
                if label != 'p':
                    for videofile in [x for x in os.listdir('CountingData/' + project + '/' + video + '/' + label) if x[0] != '.']:
                        if random.randint(0,4) == 0:
                            image_data['val']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)
                        else:
                            image_data['train']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)

    return CountingDataset(image_data['train'], data_transforms['train']), CountingDataset(image_data['val'], data_transforms['val']),

def createModel(args):
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    if not args.unfreeze:
        for param in model_ft.parameters():
            param.requires_grad = False
    if args.LossFunction == 'CE':
        model_ft.fc = nn.Linear(num_ftrs, 1)
    else:
        model_ft.fc = nn.Linear(num_ftrs, 6)
    return model_ft

parser = argparse.ArgumentParser()
parser.add_argument('LossFunction', type = str, choices=['L1','L2','CE'], help = 'Loss functions L1, L2, or cross entropy')
parser.add_argument('-u', '--unfreeze', action = 'store_true', help = 'Use if you want to fit all parameters')
parser.add_argument('-g', '--gpu', type = int, help = 'Use if you want to specify the gpu card to use')

args = parser.parse_args()

# Download data
subprocess.call(['rclone', 'copy', 'cichlidVideo:McGrath/Apps/CichlidPiData/__Counting/', 'CountingData'])

trainDataset, valDataset = prepareData()

model = createModel(args)

if args.gpu is None:
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


