import argparse, subprocess, os, random, torch, pdb
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

class CountingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict, transforms=None):
        self.data_list = list(data_dict.keys())
        self.data_dict = data_dict
        self.transforms = transforms

    def __getitem__(self, index):
       
        with open(self.data_list[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
            pdb.set_trace()

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
        model_ft.fc = nn.Linear(num_ftrs, 6)
    else:
        model_ft.fc = nn.Linear(num_ftrs, 1)
    return model_ft

parser = argparse.ArgumentParser()
parser.add_argument('LossFunction', type = str, choices=['L1','L2','CE'], help = 'Loss functions L1, L2, or cross entropy')
parser.add_argument('-u', '--unfreeze', action = 'store_true', help = 'Use if you want to fit all parameters')
parser.add_argument('-g', '--gpu', type = int, help = 'Use if you want to specify the gpu card to use')

args = parser.parse_args()

# Download data
subprocess.call(['rclone', 'copy', 'cichlidVideo:McGrath/Apps/CichlidPiData/__Counting/', 'CountingData'])

# Prepare data to create training sets
trainDataset, valDataset = prepareData()

# Create model
model = createModel(args)

# Identify GPU device to run code on
if args.gpu is None:
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

# Set criterion:
if args.LossFunction == 'L1':
    criterion = nn.L1Loss()
elif args.LossFunction == 'L2':
    criterion = nn.MSELoss()
elif args.LossFunction == 'CE':
    cweights = [0.6035, 0.6137, 0.8485, 0.9499, 0.9851, 1] #class weights for samples 3942:3841:1506:498:148:8
    class_weights = torch.FloatTensor(cweights).to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights)
else:
    raise Exception('Not sure how to handle ' + args.LossFunction)
# Set Optimizers:
#optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epoch_loss
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
