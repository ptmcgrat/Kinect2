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
    transforms.Normalize([0.25,0.25,0.25],[0.25,0.25,0.25])]),
    'val': transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.25,0.25,0.25],[0.25,0.25,0.25])])
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

    dataloaders = {x: torch.utils.data.DataLoader(CountingDataset(image_data[x]), batch_size=4, shuffle=True, num_workers=4, sampler=None) for x in ['train', 'val']}
    return dataloaders

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

def trainModel(dataloaders, model, criterion, optimizer, scheduler, num_epochs=100):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for e in range(num_epochs):
        print('Epoch {}/{}'.format(e, num_epochs - 1))
        print('-' * 10)

        #training/validation
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train() #set model to training mode
            else:
                model.eval() #set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            #iterate over data
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                #pdb.set_trace()

                #zero parameter gradients
                optimizer.zero_grad()

                #forward pass
                #track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #output = outputs[:, -1] ###only use for L1 or MSELoss
                    loss = criterion(outputs, labels) #should be labels.float() for L1 or MSELoss

                    #backward pass / optimization only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print() #empty line

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model

parser = argparse.ArgumentParser()
parser.add_argument('LossFunction', type = str, choices=['L1','L2','CE'], help = 'Loss functions L1, L2, or cross entropy')
parser.add_argument('-u', '--unfreeze', action = 'store_true', help = 'Use if you want to fit all parameters')
parser.add_argument('-g', '--gpu', type = int, help = 'Use if you want to specify the gpu card to use')

args = parser.parse_args()

# Download data
subprocess.call(['rclone', 'copy', 'cichlidVideo:McGrath/Apps/CichlidPiData/__Counting/', 'CountingData'])

# Prepare data to create training sets
dataLoaders = prepareData()

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

model = trainModel(dataLoaders, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)