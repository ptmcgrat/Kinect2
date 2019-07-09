import os, subprocess, sys, shutil, socket, getpass, datetime, pdb, pickle
import pandas as pd
import numpy as np
import scipy.special
from random import randint
from skimage import io
from collections import defaultdict, OrderedDict



class MachineLearningMaker:

    def __init__(self, modelIDs, localMasterDirectory, cloudMasterDirectory, localClipsDirectories, classIndFile):
        
        self.fnull = open(os.devnull, 'w') # for getting rid of standard error if desired

        for modelID in modelIDs:
            if modelID[0:5].lower() != 'model':
                raise Exception('modelID must start with "model", user named modelID=' + modelID)
        # Include code to check if model exists already

        self.modelIDs = modelIDs # Name of machine learning model

        # Store relevant directories for cloud and local data   
        self.cloudMasterDirectory = cloudMasterDirectory + '/' if cloudMasterDirectory[-1] != '/' else cloudMasterDirectory # Master directory

        self.localMasterDirectory = localMasterDirectory + '/' if localMasterDirectory[-1] != '/' else localMasterDirectory

        self.localClipsDirectories = localClipsDirectories

        # Create directories if necessary
        #os.makedirs(self.localClipsDirectory) if not os.path.exists(self.localClipsDirectory) else None

        # Directory containg python3 scripts for creating 3D Resnet 
        self.resnetDirectory = os.getenv("HOME") + '/3D-resrets/'
        
        # Store and download label file
        self.classIndFile = self.localMasterDirectory + classIndFile # This file lists the allowed label classes
        #print(['rclone', 'copy', self.cloudMasterDirectory + classIndFile, self.localMasterDirectory])
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + classIndFile, self.localMasterDirectory], stderr = self.fnull)
        assert os.path.exists(self.classIndFile)
        
        # Identify classes
        self.classes, self.numClasses = self._identifyClasses()

        self._print('ModelInitialization: modelIDs: ' + ','.join(modelIDs))

    def prepareData(self, labeledClusterFile = None):

        # Download and open manual label file if necessary
        if labeledClusterFile is not None:
            self.labeledData, self.numLabeledClusters = self._loadClusterFile(labeledClusterFile)
 
        # Download clips
        self._print('Converting mp4s into jpgs and creating train/test datasets', log = False)
        self._convertClips()

        # Run cichlids_json script to create json info for all clips
 

    def runTraining(self, projects, GPU = 0):
        """ Projects is a dictionary mapping the model ID to the projects that should be used"""
        #self.classes, self.numClasses = self._identifyClasses()
        # Run cichlids_json script to create json info for all clips
        #self.localOutputDirectory = self.localMasterDirectory + modelID + '/' # Where all model data will be stored
        processes = []
        for modelID in self.modelIDs:
            localModelDirectory = self.localMasterDirectory + modelID + '/'
            resultsDirectory = 'resnet18/'
            shutil.rmtree(localModelDirectory + resultsDirectory) if os.path.exists(localModelDirectory + resultsDirectory) else None
            os.makedirs(localModelDirectory + resultsDirectory)
            with open(localModelDirectory + 'cichlids_train_list.txt', 'w') as f, open(localModelDirectory + 'cichlids_test_list.txt', 'w') as g, open(localModelDirectory + 'AnnotationFile.csv', 'w') as h:
                for clipsDirectory in self.localClipsDirectories:
                    clips = [x for x in os.listdir(clipsDirectory) if '.mp4' in x]
                assert len(clips) != 0
                for clip in clips:
                    try:
                        LID,N,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:5]]
                    except IndexError: #MC6_5
                        self._print(clip)
                        LID,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:4]]
                    except ValueError:
                        self._print('ClipError: ' + str(clip))
                        LID,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:4]]

                    subTable = self.labeledData.loc[(self.labeledData.LID == LID) & (self.labeledData.t == t) & (self.labeledData.X == x) & (self.labeledData.Y == y)]
                    projectID, videoID, label = [x.values[0] for x in [subTable.projectID, subTable.videoID, subTable.ManualLabel]]

                    if projectID in projects[modelID]:
                        if randint(0,4) == 4: # Test data
                            print(label + '/' + clip.replace('.mp4',''), file = g)
                            print(clip.replace('.mp4','') + ',Test,' + label + ',' + projectID + ':' + videoID, file = h)
                        else: # Train data
                            print(label + '/' + clip.split('/')[-1].replace('.mp4',''), file = f)
                            print(clip.replace('.mp4','') + ',Train,' + label + ',' + projectID + ':' + videoID, file = h)

            command = []
            command += ['python', self.resnetDirectory + 'utils/cichlids_json.py']
            command += [localModelDirectory]
            command += [self.classIndFile]
            print(command)
            subprocess.call(command)

            self._print('modelCreation: GPU:' + str(GPU))

            command = OrderedDict()
            command['python'] = self.resnetDirectory + 'main.py'
            command['--root_path'] = self.localMasterDirectory
            command['--video_path'] = 'Clips'
            command['--annotation_path'] = modelID + '/cichlids.json'
            command['--model'] = 'resnet'
            command['--model_depth'] = '18'
            command['--n_classes'] = str(self.numClasses)
            command['--batch_size'] = '3'
            command['--n_threads'] = '5'
            command['--checkpoint'] = '5'
            command['--dataset'] = 'cichlids'
            command['--sample_duration'] = 90
            command['--sample_size'] = 260
            command['--n_epochs'] = '100'
            command['--weight_decay'] = str(1e-23)
            command['--n_val_samples'] = '1'
            command['--mean_file'] = self.localMasterDirectory + 'Means.csv'
            command['--annotation_file'] = self.localMasterDirectory + 'AnnotationFile.csv'
            command['--result_path'] = modelID + '/' + resultsDirectory
        
            trainEnv = os.environ.copy()
            trainEnv['CUDA_VISIBLE_DEVICES'] = str(GPU)
            
            subprocess.call(['cp', self.localMasterDirectory + 'Means.csv', localModelDirectory])
            subprocess.call(['cp', self.classIndFile, localModelDirectory + 'classInd.txt'])
            pickle.dump(command, open(localModelDirectory + 'commands.pkl', 'wb'))

            outCommand = []
            [outCommand.extend([str(a),str(b)]) for a,b in zip(command.keys(), command.values())]
            self._print(' '.join(outCommand))
            GPU += 1
            #processes.append(subprocess.Popen(outCommand, env = trainEnv, stdout = open(self.localOutputDirectory + resultsDirectory + 'RunningLogOut.txt', 'w'), stderr = open(self.localOutputDirectory + resultsDirectory + 'RunningLogError.txt', 'w')))
        return True

    def predictLabels(self, modelIDs, GPU = 4):


        processes = []
        for modelID in modelIDs:
            self._print('modelPrediction: GPU:' + str(GPU) + ',,modelID:' + modelID)

            cloudModelDir = self.cloudModelDirectory + modelID + '/'
            localModelDir = self.localOutputDirectory + modelID + '/'
            subprocess.call(['rclone', 'copy', cloudModelDir + 'model.pth', localModelDir])
            assert os.path.exists(localModelDir + 'model.pth')
            subprocess.call(['rclone', 'copy', cloudModelDir + 'commands.pkl', localModelDir])
            assert os.path.exists(localModelDir + 'commands.pkl')
            print(['rclone', 'copy', cloudModelDir + 'classInd.txt', self.localOutputDirectory])
            subprocess.call(['rclone', 'copy', cloudModelDir + 'classInd.txt', self.localOutputDirectory])
            assert os.path.exists(self.localOutputDirectory + 'classInd.txt')
            self.classIndFile = self.localOutputDirectory + 'classInd.txt'
            self.classes, self.numClasses = self._identifyClasses()

            command = []
            command += ['python', self.resnetDirectory + 'utils/cichlids_json.py']
            command += [self.localOutputDirectory]
            command += [self.localOutputDirectory + 'classInd.txt']
            print(command)
            subprocess.call(command)

            with open(localModelDir + 'commands.pkl', 'rb') as pickle_file:
                command = pickle.load(pickle_file) 
        
            command['--root_path'] = self.localOutputDirectory
            command['--n_epochs'] = '1'
            command['--pretrain_path'] = localModelDir + 'model.pth'
            command['--mean_file'] = self.localOutputDirectory + 'Means.csv'
            command['--annotation_file'] = self.localOutputDirectory + 'AnnotationFile.csv'

            resultsDirectory = 'prediction/'
            shutil.rmtree(localModelDir + resultsDirectory) if os.path.exists(localModelDir + resultsDirectory) else None
            os.makedirs(localModelDir + resultsDirectory) 
            trainEnv = os.environ.copy()
            trainEnv['CUDA_VISIBLE_DEVICES'] = str(GPU)
            command['--result_path'] = modelID + '/' + resultsDirectory

        #pickle.dump(command, open(self.localOutputDirectory + 'commands.pkl', 'wb'))

            outCommand = []
            [outCommand.extend([str(a),str(b)]) for a,b in zip(command.keys(), command.values())] + ['--no_train']
            print(outCommand)
            processes.append(subprocess.Popen(outCommand, env = trainEnv, stdout = open(localModelDir + resultsDirectory + 'RunningLogOut.txt', 'w'), stderr = open(localModelDir + resultsDirectory + 'RunningLogError.txt', 'w')))
            GPU += 1

        for process in processes:
            process.communicate()

        predictions = []
        for modelID in modelIDs:
            localOutDir = self.localOutputDirectory + modelID + '/prediction/'
            dt = pd.read_csv(self.localOutputDirectory + modelID + '/prediction/ConfidenceMatrix.csv', header = None, names = ['Filename'] + self.classes, skiprows = [0], index_col = 0)
            softmax = dt.apply(scipy.special.softmax, axis = 1)
            prediction = pd.concat([softmax.idxmax(axis=1).rename(modelID + '_pred'), softmax.max(axis=1).rename(modelID + '_conf')], axis=1)

            prediction['LID'] = prediction.apply(lambda row: int(row.name.split('/')[-1].split('_')[0]), axis = 1)
            prediction['N'] = prediction.apply(lambda row: int(row.name.split('/')[-1].split('_')[1]), axis = 1)

            predictions.append(prediction)

        return predictions

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

    def _loadClusterFile(self, labeledClusterFile):
        #subprocess.call(['rclone', 'copy', self.cloudModelDirectory + self.labeledClusterFile, self.localOutputDirectory], stderr = self.fnull)
        dt = pd.read_csv(labeledClusterFile, sep = ',', header = 0, index_col=0)
        #dt = dt[dt.projectID.isin(self.projects)] # Filter to only include data for projectIDs included for this model
        #dt.to_csv(self.localOutputDirectory + self.labeledClusterFile, sep = ',') # Overwrite csv file to only include this data
        #self._print('ClassDistribution:')
        #self._print(dt.groupby(['ManualLabel']).count()['LID'])
        return dt, len(dt)

    def _convertClips(self):

        means = {}

        for clipsDirectory in self.localClipsDirectories:
            self._print('Processing ' + clipsDirectory, log = False)
            clips = [x for x in os.listdir(clipsDirectory) if '.mp4' in x]
            assert len(clips) != 0
            assert os.path.exists(clipsDirectory + 'Means.npy')
            for clip in clips:
                try:
                    LID,N,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:5]]
                except IndexError: #MC6_5
                    self._print(clip)
                    LID,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:4]]
                except ValueError:
                    self._print('ClipError: ' + str(clip))
                    LID,t,x,y = [int(x) for x in clip.split('/')[-1].split('.')[0].split('_')[0:4]]

                try:
                    subTable = self.labeledData.loc[(self.labeledData.LID == LID) & (self.labeledData.t == t) & (self.labeledData.X == x) & (self.labeledData.Y == y)]
                except AttributeError:
                    projectID, videoID, label = '','',self.classes[0]
                    print(label + '/' + clip.replace('.mp4',''), file = g)
                    print(clip.replace('.mp4','') + ',Test,' + label + ',' + projectID + ':' + videoID, file = h)
                else:
                    if len(subTable) == 0:
                        raise Exception('No label for: ' + clip)
                    elif len(subTable) > 1:
                        raise Exception('Multiple labels for: ' + clip)
                    else:
                        projectID, videoID, label = [x.values[0] for x in [subTable.projectID, subTable.videoID, subTable.ManualLabel]]
            
                outDirectory = self.localMasterDirectory + 'Clips/' + label + '/' + clip.replace('.mp4','') + '/'
                shutil.rmtree(outDirectory) if os.path.exists(outDirectory) else None
                os.makedirs(outDirectory) 
                #print(['ffmpeg', '-i', self.localClipsDirectory + projectID + '/' + videoID + '/' + clip, outDirectory + 'image_%05d.jpg'])
                subprocess.call(['ffmpeg', '-i', clipsDirectory + clip, outDirectory + 'image_%05d.jpg'], stderr = self.fnull)

                frames = [x for x in os.listdir(outDirectory) if '.jpg' in x]
                try:
                    if self.nFrames != len(frames):
                        raise Exception('Different number of frames than expected in: ' + clip)
                except AttributeError:
                    self.nFrames = len(frames)

                with open(outDirectory + 'n_frames', 'w') as i:
                    print(str(self.nFrames), file = i)


            means[projectID + ':' + videoID] = np.load(clipsDirectory + 'Means.npy')

        with open(self.localMasterDirectory + 'Means.csv', 'w') as f:
            print('meanID,redMean,greenMean,blueMean,redStd,greenStd,blueStd', file = f)
            for meanID,data in means.items():
                print(meanID + ',' + ','.join([str(x) for x in list(data[0]) + list(data[1])]), file = f)
    
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
        print(outtext, file = sys.stderr)
        return
        if log:
            with open(self.localOutputDirectory + self.modelID + '_MachineLearningLog.txt', 'a') as f:
                print(str(outtext) + '...' + str(getpass.getuser()) + ', ' + str(datetime.datetime.now()) + ', ' + socket.gethostname(), file = f)
        print(outtext, file = sys.stderr)
    

