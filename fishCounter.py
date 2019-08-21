import argparse, subprocess, os, random
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

class CountingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict, transforms=None):
        self.data_list = list(data_dict.keys())
        self.data_dict = data_dict
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        


        path = self.data[index]['video']
        clip_name = path.rstrip().split('/')[-1]
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transforms is not None:
            self.spatial_transforms[self.annotationDict[clip_name]].randomize_parameters()
            clip = [self.spatial_transforms[self.annotationDict[clip_name]](img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target, path

    def __len__(self):
        return len(self.data_dict)

parser = argparse.ArgumentParser()
parser.add_argument('LossFunction', type = str, choices=['L1','L2','CE'], help = 'Loss functions L1, L2, or cross entropy')
parser.add_argument('-u', '--unfreeze', action = 'store_true', help = 'Use if you want to fit all parameters')
parser.add_argument('-g', '--gpu', type = int, help = 'Use if you want to specify the gpu card to use')

args = parser.parse_args()

# Download data
subprocess.call(['rclone', 'copy', 'cichlidVideo:McGrath/Apps/CichlidPiData/__Counting/', 'CountingData'])

image_data = {}
image_data['train'] = {}
image_data['val'] = {}

for project in [x for x in os.listdir('CountingData/') if x[0] != '.']:
    for video in [x for x in os.listdir('CountingData/' + project) if x[0] != '.']:
        for label in [x for x in os.listdir('CountingData/' + project + '/' + video) if x[0] != '.']:
            if label != 'p':
                for videofile in [x for x in os.listdir('CountingData/' + project + '/' + video + '/' + label) if x[0] != '.']:
                    if random.randint(0,4) == '0':
                        image_data['val']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)
                    else:
                        image_data['train']['CountingData/' + project + '/' + video + '/' + label + '/' + videofile] = int(label)

