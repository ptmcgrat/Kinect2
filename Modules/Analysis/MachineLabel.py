import os
from subprocess import call

class MachineLabelAnalyzer:
    def __init__(self, projectID, videoID, dataDirectory, labelFile):
        self.dataDirectory = dataDirectory
        self.projectID = projectID
        self.videoID = videoID
        self.labelFile = labelFile

        self.tempMasterDirectory = os.getenv("HOME") + '/Temp/MachinePrediction/' + projectID + '/' + videoID + '/'
        self.tempDataDirectory = self.tempMasterDirectory + 'jpgs/'
        os.makedirs(self.tempDataDirectory) if not os.path.exists(self.tempDataDirectory) else None

        self.machineLearningDirectory = os.getenv("HOME") + '/3D-ResNets-PyTorch/'

        #os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        call(['cp', dataDirectory + 'model.pth', self.tempMasterDirectory])

        self.fnull = open(os.devnull, 'w')

    def prepareData(self):
        print('Preparing data')
        with open(self.tempMasterDirectory + 'cichlids_test_list.txt', 'w') as f:
            for mp4File in os.listdir(self.dataDirectory):
                if '.mp4' not in mp4File:
                    continue
                print('m/' + mp4File.replace('.mp4',''), file = f)
                jpegDirectory = self.tempDataDirectory + 'm/' + mp4File.replace('.mp4','') + '/'
                os.makedirs(jpegDirectory) if not os.path.exists(jpegDirectory) else None
                call(['ffmpeg', '-i', self.dataDirectory + mp4File, jpegDirectory + 'image_%05d.jpg'], stderr = self.fnull)

                # Add n_frames info
                with open(jpegDirectory + 'n_frames', 'w') as g:
                    print('120', file = g)
                    
        f = open(self.tempMasterDirectory + 'cichlids_train_list.txt', 'w')
        f.close()
        
        call(['python',self.machineLearningDirectory + 'utils/cichlids_json.py', self.tempMasterDirectory, self.dataDirectory + 'classInd.txt'])       
        
    def makePredictions(self):
        print('Making predictions')
        print(' '.join(['python',self.machineLearningDirectory + 'main.py', '--root_path', self.tempMasterDirectory, '--video_path', 'jpgs', '--annotation_path', 'cichlids.json', '--result_path', 'result', '--model', 'resnet', '--model_depth', '18', '--n_classes', '7', '--batch_size', '12', '--n_threads', '5', '--dataset', 'cichlids', '--sample_duration', '120', '--mean_dataset', 'cichlids' ,'--train_crop' ,'random' ,'--n_epochs' ,'1' ,'--pretrain_path', 'model.pth' ,'--weight_decay' ,'1e-12' ,'--n_val_samples', '1' ,'--n_finetune_classes', '7', '--no_train']))
        call(['python',self.machineLearningDirectory + 'main.py', '--root_path', self.tempMasterDirectory, '--video_path', 'jpgs', '--annotation_path', 'cichlids.json', '--result_path', 'result', '--model', 'resnet', '--model_depth', '18', '--n_classes', '7', '--batch_size', '12', '--n_threads', '5', '--dataset', 'cichlids', '--sample_duration', '120', '--mean_dataset', 'cichlids' ,'--train_crop' ,'random' ,'--n_epochs' ,'1' ,'--pretrain_path', 'model.pth' ,'--weight_decay' ,'1e-12' ,'--n_val_samples', '1' ,'--n_finetune_classes', '7', '--no_train'])
        
