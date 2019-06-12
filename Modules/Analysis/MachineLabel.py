import os, subprocess, sys, shutil
import pandas as pd
from random import randint


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
                subprocess.call(['ffmpeg', '-i', self.dataDirectory + mp4File, jpegDirectory + 'image_%05d.jpg'], stderr = self.fnull)

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
        
class MachineLabelCreator:
    def __init__(self, modelID, projects, localMasterDirectory, cloudMasterDirectory, labeledClusterFile, classIndFile):
        self.modelID = modelID
        self.projects = projects
        self.localMasterDirectory = localMasterDirectory
        self.cloudMasterDirectory = cloudMasterDirectory
        self.localClipsDirectory = localMasterDirectory + 'Clips/'
        self.cloudClipsDirectory = cloudMasterDirectory + 'Clips/'
        self.localJpegDirectory = self.localMasterDirectory + 'jpgs/'
        self.labeledClusterFile = labeledClusterFile
        self.classIndFile = classIndFile

        self.machineLearningDirectory = os.getenv("HOME") + '/3D-ResNets-PyTorch/'

        self.fnull = open(os.devnull, 'w')

        os.makedirs(self.localMasterDirectory) if not os.path.exists(self.localMasterDirectory) else None
        os.makedirs(self.localJpegDirectory) if not os.path.exists(self.localJpegDirectory) else None

    def prepareData(self):
        allClips = set()
        print('Downloading ' + self.labeledClusterFile, file = sys.stderr)
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.labeledClusterFile, self.localMasterDirectory], stderr = self.fnull)
        self.dt = pd.read_csv(self.localMasterDirectory + self.labeledClusterFile, sep = ',')

        print('Creating: ' + self.localMasterDirectory + 'cichlids_train_list.txt', file = sys.stderr)
        with open(self.localMasterDirectory + 'cichlids_train_list.txt', 'w') as f, open(self.localMasterDirectory + 'cichlids_test_list.txt', 'w') as g:

            for projectID in self.projects:
                print('Downloading clips from: ' + self.cloudClipsDirectory + projectID, file = sys.stderr)
                #subprocess.call(['rclone', 'copy', self.cloudClipsDirectory + projectID, self.localClipsDirectory + projectID], stderr = self.fnull)
                videos = os.listdir(self.localClipsDirectory + projectID)

                for videoID in sorted(videos):
                    clips = os.listdir(self.localClipsDirectory + projectID + '/' + videoID)
                    for clip in clips:
                        if 'mp4' not in clip:
                            continue
                        LID = int(clip.split('_')[0])
                        try:
                            label = self.dt.loc[(self.dt.projectID == projectID) & (self.dt.videoID == videoID) & (self.dt.LID == LID)]['ManualLabel'].values[0]
                        except IndexError:
                            print('LabelError: ' + projectID + ' ' + videoID + str(LID))
                            continue
                            raise Exception('No Label for ' + clip)

                        if randint(0,4) == 4:
                            print(label + '/' + clip.replace('.mp4',''), file = g)
                        else:
                            print(label + '/' + clip.replace('.mp4',''), file = f)
                            
                        outDirectory = self.localJpegDirectory + label + '/' + clip.replace('.mp4','') + '/'
                        shutil.rmtree(outDirectory) if os.path.exists(outDirectory) else None
                        os.makedirs(outDirectory) 
                        #print(['ffmpeg', '-i', self.localClipsDirectory + projectID + '/' + videoID + '/' + clip, outDirectory + 'image_%05d.jpg'])
                        subprocess.call(['ffmpeg', '-i', self.localClipsDirectory + projectID + '/' + videoID + '/' + clip, outDirectory + 'image_%05d.jpg'], stderr = self.fnull)
                        with open(outDirectory + 'n_frames', 'w') as h:
                            print('120', file = h)

                        allClips.add((projectID, videoID, int(clip.split('_')[0])))

    def runTraining(self):
        command = []
        command += ['python', self.machineLearningDirectory + 'utils/cichlids_json.py']
        command += [self.classIndFile]
        print(command)
        subprocess.call(command)
        command = []

        command += ['python',self.machineLearningDirectory + 'main.py']
        command += ['--root_path', self.localMasterDirectory]
        command += ['--video_path', 'jpgs']
        command += ['--annotation_path', 'cichlids.json']
        command += ['--result_path', 'result']
        command += ['--model', 'resnet'] 
        command += ['--model_depth', '18'] 
        command += ['--n_classes', '7'] 
        command += ['--batch_size', '12']
        command += ['--n_threads', '5']
        command += ['--dataset', 'cichlids']
        command += ['--sample_duration', '120']
        command += ['--mean_dataset', 'cichlids']
        command += ['--train_crop' ,'random']
        command += ['--n_epochs' ,'1'] 
        command += ['--pretrain_path', 'model.pth']
        command += ['--weight_decay' ,'1e-12']
        command += ['--n_val_samples', '1']
        command += ['--n_finetune_classes', '7']
        command += ['--no_train']
        print(command)
        subprocess.call(command)

      

