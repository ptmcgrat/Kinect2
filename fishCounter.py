import argparse, subprocess, os, random, torch, pdb, time, copy
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


class FishCounter:
    def __init__(self, dataLoader, lossFunction, modelDepth, lastLayerFlag, optimizer, device=2, lr = 0.001):

        self.device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

        self.prepareData(dataLoader)
        self.createModel(lossFunction, modelDepth, lastLayerFlag)
        self.createCriterion(lossFunction)
        self.setOptimizer(optimizer, lr)
        
        
    def prepareData(self, command):
        commands = ['Normal']
        if command not in commands:
            raise ValueException('command argument must be one of ' + ','.join(commands))

        if command == 'Normal':
            self._dataLoaderNormal()

    def _dataLoaderNormal(self):
        
        # Copy data from Dropbox
        subprocess.call(['rclone', 'copy', 'cichlidVideo:McGrath/Apps/CichlidPiData/__Counting/', 'CountingData'])
        
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
                        for videofile in [x for x in os.listdir('CountingData/' + project + '/' + video + '/' + label) if x[-3:] == 'jpg']:
                            if random.randint(0,4) == 0:
                                image_data['val']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)
                            else:
                                image_data['train']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)

        self.dataloaders = {x: torch.utils.data.DataLoader(CountingDataset(image_data[x], transforms = data_transforms[x]), batch_size=48, shuffle=True, num_workers=4, sampler=None) for x in ['train', 'val']}
        
    def createModel(self, command, depth, lastLayerFlag):
        commands = ['L1', 'L2', 'CE']
        if command not in commands:
            raise ValueException('command argument must be one of ' + ','.join(commands))

        depth = str(depth) # in case integer value used
        depths = ['10', '18', '34', '50']
        if depth not in depths:
            raise ValueException('command argument must be one of ' + ','.join(depths))

        if depth == '10':
            model = models.resnet10(pretrained=True)
        if depth == '18':
            model = models.resnet18(pretrained=True)
        if depth == '34':
            model = models.resnet34(pretrained=True)
        if depth == '50':
            model = models.resnet50(pretrained=True)
   
        num_ftrs = model.fc.in_features

        if lastLayerFlag:
            for param in model.parameters():
                param.requires_grad = False

        if command == 'CE':
            model.fc = nn.Linear(num_ftrs, 6)
        else:
            model.fc = nn.Linear(num_ftrs, 1)

        self.model = model

    def createCriterion(self, command):
        commands = ['L1', 'L2', 'CE']
        if command not in commands:
            raise ValueException('command argument must be one of ' + ','.join(commands))

        if command == 'L1':
            self.criterion = nn.L1Loss()
        elif command == 'L2':
            self.criterion = nn.MSELoss()
        elif command == 'CE':
            cweights = [0.6035, 0.6137, 0.8485, 0.9499, 0.9851, 1] #class weights for samples 3942:3841:1506:498:148:8
            class_weights = torch.FloatTensor(cweights).to(device)
            criterion = nn.CrossEntropyLoss(weight = class_weights)

    def setOptimizer(self, command, lr = 0.001):
        commands = ['adam', 'sgd']
        if command not in commands:
            raise ValueException('command argument must be one of ' + ','.join(commands))

        if command == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

        if command == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def setScheduler(self, command, **kwargs):
        commands = ['none', 'step', 'cycle']
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def trainModel(self, num_epochs = 100):
    
        since = time.time()
        self.model = self.model.to(self.device)
        #best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for e in range(num_epochs):
            print('Epoch {}/{}'.format(e, num_epochs - 1))
            print('-' * 10)

            #training/validation
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.lr_scheduler.step()
                    self.model.train() #set model to training mode
                else:
                    self.model.eval() #set model to evaluation mode

                running_loss,running_corrects = 0.0,0

                #iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    #pdb.set_trace()

                    #zero parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    #track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)

                        if type(self.criterion) == torch.nn.modules.CrossEntropyLoss:
                            loss = self.criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                        else:
                            loss = self.criterion(outputs[:,-1], labels.float())
                            preds = outputs.int()[:,-1].type(torch.int64)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    #stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


                epoch_loss = running_loss / len(self.dataloaders[phase])
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                #deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_model_wts = copy.deepcopy(model.state_dict())

        print() #empty line

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val accuracy: {:.4f}'.format(best_acc))

        #model.load_state_dict(best_model_wts)




parser = argparse.ArgumentParser()
#parser.add_argument('LossFunction', type = str, choices=['L1','L2','CE'], help = 'Loss functions L1, L2, or cross entropy')
#parser.add_argument('-u', '--unfreeze', action = 'store_true', help = 'Use if you want to fit all parameters')
#parser.add_argument('-g', '--gpu', type = int, help = 'Use if you want to specify the gpu card to use')

args = parser.parse_args()

fc = FishCounter('Normal', 'L1', '50', True, 'adam', 0, 0.001)