import os, subprocess, sys, shutil, socket, getpass, datetime
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

        if modelID[0:5].lower() != 'model':
            raise Exception('modelID must start with "model", user named modelID=' + modelID)
        # Include code to check if model exists already

        self.modelID = modelID # Name of machine learning model
        self.projects = projects # Projects that will be included in this data

        # Store relevant directories for cloud and local data   
        self.cloudMasterDirectory = cloudMasterDirectory + '/' if cloudMasterDirectory[-1] != '/' else cloudMasterDirectory # Master directory
        self.cloudOutputDirectory = self.cloudMasterDirectory + '/' + modelID + '/' # Where data will be stored once model is trained
        self.cloudClipsDirectory = self.cloudOutputDirectory + 'Clips/' # Where manually labeled clips are stored

        self.localMasterDirectory = localMasterDirectory + '/' if localMasterDirectory[-1] != '/' else localMasterDirectory
        self.localOutputDirectory = self.localMasterDirectory + '/' + modelID + '/' # Where all model data will be stored
        self.localClipsDirectory = self.localOutputDirectory + 'Clips/' # Where mp4 clips and created jpg images will be stored

        # Create directories if necessary
        os.makedirs(self.localClipsDirectory) if not os.path.exists(self.localClipsDirectory) else None

        # Directory containg python3 scripts for creating 3D Resnet 
        self.resnetDirectory = os.getenv("HOME") + '/3D-ResNets-PyTorch/'

        # Store file names
        self.labeledClusterFile = 'ManualLabeledClusters.csv' # This file that contains the manual label information for each clip
        self.classIndFile = classIndFile # This file lists the allowed label classes

        self.fnull = open(os.devnull, 'w') # for getting rid of standard error if desired

        self._print('ModelCreation: modelID: ' + modelID + ',,projectsUsed:' + ','.join(projects))

    def prepareData(self):

        # Determine how many label classes are possible
        self.classes, self.numClasses = self._identifyClasses()

        # Download and open manual label file
        self.labeledData, self.numLabeledClusters = self._loadClusterFile()
 
        # Download clips
        for projectID in self.projects:
            self._print('Downloading clips for ' + projectID + ' from ' + self.cloudClipsDirectory + projectID, log=False)
            subprocess.call(['rclone', 'copy', self.cloudClipsDirectory + projectID, self.localClipsDirectory], stderr = self.fnull)

        self._print('Converting mp4s into jpgs and creating train/test datasets', log = False)
        self._convertClips()

        # Run cichlids_json script to create json info for all clips
        command = []
        command += ['python', self.machineLearningDirectory + 'utils/cichlids_json.py']
        command += [self.localOutputDirectory]
        command += [self.classIndFile]
        print(command)
        subprocess.call(command)

    def runTraining(self):

        self.resultDirectories = []

        processes = []
        for i in range(1):
            weightDecay = 10**(-1*(23-i))
            print(weightDecay)

            resultsDirectory = str(weightDecay) + '/'
            self.resultDirectories.append(resultsDirectory)
            shutil.rmtree(self.localMasterDirectory + resultsDirectory) if os.path.exists(self.localMasterDirectory + resultsDirectory) else None
            os.makedirs(self.localMasterDirectory + resultsDirectory) if not os.path.exists(self.localMasterDirectory + resultsDirectory) else None
            trainEnv = os.environ.copy()
            #trainEnv['CUDA_VISIBLE_DEVICES'] = str(i)
            print(trainEnv['CUDA_VISIBLE_DEVICES'])

            command = []
            command += ['python',self.machineLearningDirectory + 'main.py']
            command += ['--root_path', self.localMasterDirectory]
            command += ['--video_path', 'jpgs']
            command += ['--annotation_path', 'cichlids.json']
            command += ['--result_path', resultsDirectory]
            command += ['--model', 'resnet'] 
            command += ['--model_depth', '18'] 
            command += ['--n_classes', '7'] 
            command += ['--batch_size', '6']
            command += ['--n_threads', '5']
            command += ['--checkpoint', '5']
            command += ['--dataset', 'cichlids']
            command += ['--sample_duration', '120']
            command += ['--mean_dataset', 'cichlids']
            command += ['--train_crop' ,'random']
            command += ['--n_epochs' ,'100'] 
            command += ['--weight_decay' , str(weightDecay)]
            command += ['--n_val_samples', '1']
            print(command)
            processes.append(subprocess.Popen(command, env = trainEnv, stdout = open(self.localMasterDirectory + resultsDirectory + 'RunningLogOut.txt', 'w'), stderr = open(self.localMasterDirectory + resultsDirectory + 'RunningLogError.txt', 'w')))
      
        for process in processes:
            process.communicate()

    def summarizeResults(self):

        for rD in self.resultDirectories:
            with open(self.localMasterDirectory + rD + 'val.log') as f:
                for line in f:
                    epoch = line.split()[0]
                    accuracy = float(line.split('tensor(')[-1].split(',')[0])

    def _identifyClasses(self):
        classes = set()
        with open(self.classIndFile) as f:
            for line in f:
                classes.add(line.rstrip().split()[-1])
        self._print('ModelCreation: numClasses: ' + str(len(classes)) + ',,ClassIDs: ' + ','.join(sorted(list(classes))))
        return classes, len(classes)

    def _loadClusterFile(self):
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.labeledClusterFile, self.localOutputDirectory], stderr = self.fnull)
        dt = pd.read_csv(self.localOutputDirectory + self.labeledClusterFile, sep = ',', header = 0, index_col=0)
        dt = dt[dt.projectID.isin(self.projects)] # Filter to only include data for projectIDs included for this model
        dt.to_csv(self.localOutputDirectory + self.labeledClusterFile, sep = ',') # Overwrite csv file to only include this data
        return dt, len(dt)

    def _convertClips(self):

        clips = []
        for projectID in self.projects:
            for videoID in os.listdir(self.localClipsDirectory + projectID):
                clips.extend([projectID + '/' + videoID + '/' + x for x in os.listdir(self.localClipsDirectory + projectID + '/' + videoID + '/') if '.mp4' in x])
        
        if len(clips) != self.numLabeledClusters:
            raise Exception('The number of clips, ' + str(len(clips)) + ', does not match the number of labeled clusters, ' + str(self.numLabeledClusters))

        self._print('ModelCreation: labeledClusters: ' + str(self.numLabeledClusters))

        means = np.array(shape = (len(clips),3))
        stds = np.array(shape = (len(clips),3))


        with open(self.localMasterDirectory + 'cichlids_train_list.txt', 'w') as f, open(self.localMasterDirectory + 'cichlids_test_list.txt', 'w') as g:
            for clip in clips:
                LID,N,t,x,y = [int(x) for x in clip.split('/')[-1].split('_')[0:5]]
                subTable = self.labeledData.loc[(self.labeledData.LID == LID) & (self.labeledData.N == N) & (self.labeledData.t == t) & (self.labeledData.x == x) & (self.labeledData.y == y)]
                if len(subTable) == 0:
                    raise Exception('No label for: ' + clip)
                elif len(subTable) > 1:
                    raise Exception('Multiple labels for: ' + clip)
                else:
                    label = subTable.values[0]

                if randint(0,4) == 4: # Test data
                    print(label + '/' + clip.replace('.mp4',''), file = g)
                else: # Train data
                    print(label + '/' + clip.replace('.mp4',''), file = f)
                            
                outDirectory = self.localClipsDirectory + label + '/' + clip.split('/')[-1].replace('.mp4','') + '/'
                shutil.rmtree(outDirectory) if os.path.exists(outDirectory) else None
                os.makedirs(outDirectory) 
                #print(['ffmpeg', '-i', self.localClipsDirectory + projectID + '/' + videoID + '/' + clip, outDirectory + 'image_%05d.jpg'])
                subprocess.call(['ffmpeg', '-i', self.localClipsDirectory + clip, outDirectory + 'image_%05d.jpg'], stderr = self.fnull)

                frames = [x for x in os.listdir(outDirectory) if '.jpg' in x]
                try:
                    if self.nFrames != len(frames):
                        raise Exception('Different number of frames than expected in: ' + clip)
                except AttributeError:
                    self.nFrames = len(frames)

                for i, frame in enumerate(frames):
                    img = io.imread(frame)
                    means[i] = img.mean(axis = (0,1))
                    stds[i] = img.std(axis = (0,1))

                with open(outDirectory + 'n_frames', 'w') as h:
                    print(str(self.nFrames, file = h))

        print(means.mean(axis = 0))
        print(stds.mean(axis = 0))

    def _print(self, outtext, log = True):
        if log:
            with open(self.localOutputDirectory + self.modelID + '_CreationLog.txt', 'a') as f:
                print(outtext + '...' + str(getpass.getuser()) + ', ' + str(datetime.datetime.now()) + ', ' + socket.gethostname(), file = f)
        print(outtext, file = sys.stderr)
    

