import os, subprocess, sys, shutil, socket, getpass, datetime, pdb, pickle
import pandas as pd
import numpy as np
from random import randint
from skimage import io
from collections import defaultdict, OrderedDict


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


class MachineLearningMaker:
    def __init__(self, modelID, projects, localMasterDirectory, cloudModelDirectory, cloudClipsDirectory, labeledClusterFile = None, classIndFile = None):

        if modelID[0:5].lower() != 'model':
            raise Exception('modelID must start with "model", user named modelID=' + modelID)
        # Include code to check if model exists already

        self.modelID = modelID # Name of machine learning model
        self.projects = projects # Projects that will be included in this data

        # Store relevant directories for cloud and local data   
        self.cloudModelDirectory = cloudModelDirectory + '/' if cloudModelDirectory[-1] != '/' else cloudModelDirectory # Master directory
        self.cloudClipsDirectory = cloudClipsDirectory + '/' if cloudClipsDirectory[-1] != '/' else cloudClipsDirectory # Where manually labeled clips are stored

        self.localMasterDirectory = localMasterDirectory + '/' if localMasterDirectory[-1] != '/' else localMasterDirectory
        self.localOutputDirectory = self.localMasterDirectory + modelID + '/' # Where all model data will be stored
        self.localClipsDirectory = self.localOutputDirectory + 'Clips/' # Where mp4 clips and created jpg images will be stored

        # Create directories if necessary
        os.makedirs(self.localClipsDirectory) if not os.path.exists(self.localClipsDirectory) else None

        # Directory containg python3 scripts for creating 3D Resnet 
        self.resnetDirectory = os.getenv("HOME") + '/3D-resnets/'

        # Store file names
        self.labeledClusterFile = labeledClusterFile # This file that contains the manual label information for each clip
        
        if classIndFile is None: # Try to download it from model directory
            subprocess.call(['rclone', 'copy', self.cloudModelDirectory + 'classInd.txt', self.localOutputDirectory])
            assert os.path.exists(self.localModelDirectory + 'classInd.txt')
            self.classIndFile = self.localModelDirectory + 'classInd.txt'

        else:
            self.classIndFile = classIndFile # This file lists the allowed label classes

        self.fnull = open(os.devnull, 'w') # for getting rid of standard error if desired

        self._print('ModelInitialization: modelID: ' + modelID + ',,projectsUsed:' + ','.join(projects))

    def prepareData(self):

        # Determine how many label classes are possible from classIndFile
        self.classes, self.numClasses = self._identifyClasses()

        # Download and open manual label file if necessary
        if self.labeledClusterFile is not None:
            self.labeledData, self.numLabeledClusters = self._loadClusterFile()
 
        # Download clips
        for projectID in self.projects:
            self._print('Downloading clips for ' + projectID + ' from ' + self.cloudClipsDirectory + projectID, log=False)
            subprocess.call(['rclone', 'copy', self.cloudClipsDirectory + projectID, self.localClipsDirectory + projectID], stderr = self.fnull)

        self._print('Converting mp4s into jpgs and creating train/test datasets', log = False)
        self._convertClips()

        # Run cichlids_json script to create json info for all clips
        command = []
        command += ['python', self.resnetDirectory + 'utils/cichlids_json.py']
        command += [self.localOutputDirectory]
        command += [self.classIndFile]
        print(command)
        subprocess.call(command)

    def runTraining(self, GPU = 0):
        #self.classes, self.numClasses = self._identifyClasses()
        # Run cichlids_json script to create json info for all clips
       
        self._print('modelCreation: GPU:' + str(GPU))

        processes = []

        command = OrderedDict()
        command['python'] = self.resnetDirectory + 'main.py'
        command['--root_path'] = self.localOutputDirectory
        command['--video_path'] = 'Clips'
        command['--annotation_path'] = 'cichlids.json'
        command['--model'] = 'resnet'
        command['--model_depth'] = '18'
        command['--n_classes'] = str(self.numClasses)
        command['--batch_size'] = '4'
        command['--n_threads'] = '5'
        command['--checkpoint'] = '5'
        command['--dataset'] = 'cichlids'
        command['--sample_duration'] = 90
        command['--sample_size'] = 200
        command['--n_epochs'] = '100'
        command['--weight_decay'] = str(1e-23)
        command['--n_val_samples'] = '1'
        command['--mean_file'] = self.localOutputDirectory + 'Means.csv'
        command['--annotation_file'] = self.localOutputDirectory + 'AnnotationFile.csv'

        resultsDirectory = 'resnetF_'+ str(GPU) + '/'
        shutil.rmtree(self.localOutputDirectory + resultsDirectory) if os.path.exists(self.localOutputDirectory + resultsDirectory) else None
        os.makedirs(self.localOutputDirectory + resultsDirectory)
        trainEnv = os.environ.copy()
        trainEnv['CUDA_VISIBLE_DEVICES'] = str(GPU)
        command['--result_path'] = resultsDirectory

        pickle.dump(command, open(resultsDirectory + 'commands.pkl', 'wb'))

        outCommand = []
        [outCommand.extend([str(a),str(b)]) for a,b in zip(command.keys(), command.values())]
        print(outCommand)
        subprocess.Popen(outCommand, env = trainEnv, stdout = open(self.localOutputDirectory + resultsDirectory + 'RunningLogOut.txt', 'w'), stderr = open(self.localOutputDirectory + resultsDirectory + 'RunningLogError.txt', 'w'))

        #subprocess.call(['cp', self.localOutputDirectory + resultsDirectory + 'save_200.pth', self.localOutputDirectory + model.pth)])
        #subprocess.call(['cp', self.localOutputDirectory + resultsDirectory + 'commands.pkl', self.localOutputDirectory)])

        #self._summarizeModel()

        
        GPU += 1
        command['--sample_duration'] = 90
        command['--sample_size'] = 250
        command['--batch_size'] = '3'

        resultsDirectory = 'resnetF_'+ str(GPU) + '/'
        shutil.rmtree(self.localOutputDirectory + resultsDirectory) if os.path.exists(self.localOutputDirectory + resultsDirectory) else None
        os.makedirs(self.localOutputDirectory + resultsDirectory) if not os.path.exists(self.localOutputDirectory + resultsDirectory) else None
        trainEnv = os.environ.copy()
        trainEnv['CUDA_VISIBLE_DEVICES'] = str(GPU)
        command['--result_path'] = resultsDirectory

        pickle.dump(command, open(resultsDirectory + 'commands.pkl', 'wb'))

        outCommand = []
        [outCommand.extend([str(a),str(b)]) for a,b in zip(command.keys(), command.values())]
        print(outCommand)
        subprocess.Popen(outCommand, env = trainEnv, stdout = open(self.localOutputDirectory + resultsDirectory + 'RunningLogOut.txt', 'w'), stderr = open(self.localOutputDirectory + resultsDirectory + 'RunningLogError.txt', 'w'))

        #for process in processes:
        #    process.communicate()"""

    def predictLabels(self, GPU = 0):

        self._print('modelPrediction: GPU:' + str(GPU))

        subprocess.call(['rclone', 'copy', self.cloudModelDirectory + 'model.pth', self.localOutputDirectory])
        subprocess.call(['rclone', 'copy', self.cloudModelDirectory + 'commands.pkl', self.localOutputDirectory])

        command = pkl.load(self.localOutputDirectory + 'commands.pkl') 
        command['--root_path'] = self.localOutputDirectory
        command['--n_epochs'] = '1'
        command['--notrain'] = ''
        command['--pretrain_path'] = self.localOutputDirectory + 'model.pth'

        resultsDirectory = 'prediction'+ str(GPU) + '/'
        shutil.rmtree(self.localOutputDirectory + resultsDirectory) if os.path.exists(self.localOutputDirectory + resultsDirectory) else None
        os.makedirs(self.localOutputDirectory + resultsDirectory) if not os.path.exists(self.localOutputDirectory + resultsDirectory) else None
        trainEnv = os.environ.copy()
        trainEnv['CUDA_VISIBLE_DEVICES'] = str(GPU)
        command['--result_path'] = resultsDirectory

        pickle.dump(command, open(self.localOutputDirectory + 'commands.pkl', 'wb'))

        outCommand = []
        [outCommand.extend([str(a),str(b)]) for a,b in zip(command.keys(), command.values())]
        print(outCommand)
        subprocess.call(outCommand, env = trainEnv, stdout = open(self.localOutputDirectory + resultsDirectory + 'RunningLogOut.txt', 'w'), stderr = open(self.localOutputDirectory + resultsDirectory + 'RunningLogError.txt', 'w'))


    def summarizeResults(self):

        for rD in self.resultDirectories:
            with open(self.localMasterDirectory + rD + 'val.log') as f:
                for line in f:
                    epoch = line.split()[0]
                    accuracy = float(line.split('tensor(')[-1].split(',')[0])

    def _identifyClasses(self):
        classes = []
        with open(self.classIndFile) as f:
            for line in f:
                tokens = line.rstrip().split()
                classes.append(tokens[1])
        self._print('ModelInitialization: numClasses: ' + str(len(classes)) + ',,ClassIDs: ' + ','.join(sorted(classes)))
        return classes, len(classes)

    def _loadClusterFile(self):
        subprocess.call(['rclone', 'copy', self.cloudModelDirectory + self.labeledClusterFile, self.localOutputDirectory], stderr = self.fnull)
        dt = pd.read_csv(self.localOutputDirectory + self.labeledClusterFile, sep = ',', header = 0, index_col=0)
        dt = dt[dt.projectID.isin(self.projects)] # Filter to only include data for projectIDs included for this model
        dt.to_csv(self.localOutputDirectory + self.labeledClusterFile, sep = ',') # Overwrite csv file to only include this data
        self._print('ClassDistribution:')
        self._print(dt.groupby(['ManualLabel']).count()['LID'])
        return dt, len(dt)

    def _convertClips(self):

        clips = defaultdict(list)
        means = {}

        for projectID in self.projects:
            if projectID == '':
                videoID = ''
                clips[projectID].extend([x for x in os.listdir(self.localClipsDirectory + 'ClusterData/Clips/') if '.mp4' in x])
                print(['rclone', 'copy', self.cloudClipsDirectory + projectID + '/' + videoID + '/' + 'Means.npy', self.localOutputDirectory])
                subprocess.call(['rclone', 'copy', self.cloudClipsDirectory + 'Means.npy', self.localOutputDirectory])
                means[projectID + ':' + videoID] = np.load(self.localOutputDirectory + 'Means.npy')

            else:
                for videoID in os.listdir(self.localClipsDirectory + projectID):
                    clips[projectID].extend([projectID + '/' + videoID + '/' + x for x in os.listdir(self.localClipsDirectory + projectID + '/' + videoID + '/') if '.mp4' in x])
                    print(['rclone', 'copy', self.cloudClipsDirectory + projectID + '/' + videoID + '/' + 'Means.npy', self.localOutputDirectory])
                    subprocess.call(['rclone', 'copy', self.cloudClipsDirectory + projectID + '/' + videoID + '/' + 'Means.npy', self.localOutputDirectory])
                    means[projectID + ':' + videoID] = np.load(self.localOutputDirectory + 'Means.npy')

        with open(self.localOutputDirectory + 'Means.csv', 'w') as f:
            print('meanID,redMean,greenMean,blueMean,redStd,greenStd,blueStd', file = f)
            for meanID,data in means.items():
                print(meanID + ',' + ','.join([str(x) for x in list(data[0]) + list(data[1])]), file = f)

        try:
            if sum(len(x) for x in clips.values()) != self.numLabeledClusters:
                self._print('Warning: The number of clips, ' + str(sum(len(x) for x in clips.values())) + ', does not match the number of labeled clusters, ' + str(self.numLabeledClusters))
        except AttributeError: # no manual label cluster file
            pass

        self._print('ModelInitialization: # of clips: ' + str(sum(len(x) for x in clips.values())))

        with open(self.localOutputDirectory + 'cichlids_train_list.txt', 'w') as f, open(self.localOutputDirectory + 'cichlids_test_list.txt', 'w') as g, open(self.localOutputDirectory + 'AnnotationFile.csv', 'w') as h:
            print('Location,Dataset,Label,meanID', file = h)
            for projectID in clips:
                outDirectories = []
                for i,clip in enumerate(clips[projectID]):
                    if i%100 == 0:
                        self._print('Processed ' + str(i) + ' videos from ' + projectID, log = False)
                    try:
                        LID,N,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:5]]
                    except IndexError: #MC6_5
                        LID,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:4]]

                    try:
                        subTable = self.labeledData.loc[(self.labeledData.LID == LID) & (self.labeledData.t == t) & (self.labeledData.X == x) & (self.labeledData.Y == y)]['ManualLabel']
                        if len(subTable) == 0:
                            raise Exception('No label for: ' + clip)
                        elif len(subTable) > 1:
                            raise Exception('Multiple labels for: ' + clip)
                        else:
                            label = subTable.values[0]
                        if randint(0,4) == 4: # Test data
                            print(label + '/' + clip.split('/')[-1].replace('.mp4',''), file = g)
                            print(clip.split('/')[-1].replace('.mp4','') + ',Test,' + label + ',' + clip.split('/')[0] + ':' + clip.split('/')[1], file = h)
                        else: # Train data
                            print(label + '/' + clip.split('/')[-1].replace('.mp4',''), file = f)
                            print(clip.split('/')[-1].replace('.mp4','') + ',Train,' + label + ',' + clip.split('/')[0] + ':' + clip.split('/')[1], file = h)
                     
                    except AttributeError: # we will be predicting labels, not creating a model
                        label = 'z'
                        print(label + '/' + clip.split('/')[-1].replace('.mp4',''), file = g)
                        print(clip.split('/')[-1].replace('.mp4','') + ',Test,' + label + ',' + clip.split('/')[0] + ':' + clip.split('/')[1], file = h)
                            
                    outDirectory = self.localClipsDirectory + label + '/' + clip.split('/')[-1].replace('.mp4','') + '/'
                    outDirectories.append(outDirectory)
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

                    with open(outDirectory + 'n_frames', 'w') as i:
                        print(str(self.nFrames), file = i)
                # Normalize imgages using mean and std
                #self._print('ModelCreation: ' + projectID + '_Means: ' + ','.join([str(x) for x in means.mean(axis=0)]) + ',,' + projectID + '_Stds: ' + ','.join([str(x) for x in stds.mean(axis=0)]))
                """for i,outDirectory in enumerate(outDirectories):
                    if i%100 == 0:
                        self._print('Normalized ' + str(i) + ' videos from ' + projectID, log = False)
                    frames = [x for x in os.listdir(outDirectory) if '.jpg' in x]
                    for frame in frames:
                        img = io.imread(outDirectory + frames[0])
                        norm = (img - mean)/(std/30) + 125
                        norm[norm < 0] = 0
                        norm[norm > 255] = 255
                        io.imsave(outDirectory + frames[0], norm.astype('uint8'))
                """
    
    def _summarizeModel(self):
        with open(self.localOutputDirectory + 'ConfusionMatrix.csv') as f, open(self.localOutputDirectory + 'ConfusionMatrixHeaders.csv', 'w') as g:
            headers = ['']
            line = next(f)
            for token in line.rstrip().split(',')[1:]:
                headers.append(self.classes[int(token)])
            print(','.join(headers), file = g)
            for line in f:
                print(self.classes[int(line.split(',')[0])] + ','.join(line.rstrip().split(',')[1:]), file = g)

        subprocess.call(['cp', self.classIndFile, self.localOutputDirectory])

        subprocess.call(['rclone', 'copy', self.localOutputDirectory, self.cloudModelDirectory])

    def _print(self, outtext, log = True):
        if log:
            with open(self.localOutputDirectory + self.modelID + '_MachineLearningLog.txt', 'a') as f:
                print(str(outtext) + '...' + str(getpass.getuser()) + ', ' + str(datetime.datetime.now()) + ', ' + socket.gethostname(), file = f)
        print(outtext, file = sys.stderr)
    

