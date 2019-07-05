import argparse, os, socket, sys, subprocess
from subprocess import call
from collections import defaultdict

class ProjectData:
    def __init__(self, excelFile, projects, machLearningIDs = None):
        excelProjects = set()
        if machLearningIDs is None:
            dt = pd.read_excel(excelFile, header = 1, converters={'ClusterAnalysis':str,'ManualPrediction':str, 'Nvideos': int})
        else:
            converters = {'ClusterAnalysis':str,'ManualPrediction':str}
            for key in machLearningIDs:
                converters[key] = str
            dt = pd.read_excel(excelFile, header = 1, converters=converters)
        self.projects = {}
        self.clusterData = {}
        self.manPredData = {}
        self.mLearningData = defaultdict(list)
        self.nVideos = {}
        for row in dt.iterrows():
            if row[1].Include == True:

                projectID = row[1].ProjectID
                self.nVideos[projectID] = row[1].Nvideos
                excelProjects.add(projectID)
                if projects is not None and projectID not in projects:
                    continue
                groupID = row[1].GroupID
                self.projects[projectID] = groupID
                try:
                    self.clusterData[projectID] = [int(x) for x in str(row[1].ClusterAnalysis).split(',')]
                except ValueError:
                    pass
                try:
                    self.manPredData[projectID] = [int(x) for x in str(row[1].ManualPrediction).split(',')]
                except ValueError:
                    pass
                if machLearningIDs is not None:
                    try:
                        for mlID in machLearningIDs:
                            if str(row[1][mlID]).lower() == 'true':
                                self.mLearningData[mlID].append(projectID)
                    except KeyError:
                        raise KeyError(mlID + ' not a column in Excel data file')
        if machLearningIDs is None and projects is not None:
            for projectID in projects:
                if projectID not in excelProjects:
                        print('Cant find projectID: ' + projectID)
                        print('Options are ' ','.join(projects))


rcloneRemote = 'cichlidVideo' #The name of the rclone remote that has access to the dropbox or other cloud account
cloudMasterDirectory = 'McGrath/Apps/CichlidPiData/' # The master directory on the cloud account that stores video and depth data
localMasterDirectory = os.getenv('HOME') + '/Temp/CichlidAnalyzer/' #
machineLearningDirectory = '__MachineLearning/'
countingDirectory = '__Counting/'
manualLabelFile = 'ManualLabeledClusters.csv' # The name of the file that contains info on all manually annotated data

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

trackerParser = subparsers.add_parser('CollectData', help='This command runs on Raspberry Pis to collect depth and RGB data')

depthParser = subparsers.add_parser('DepthAnalysis', help='This command runs depth analysis for a project')
depthParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
depthParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
depthParser.add_argument('-i', '--InitialPrep', action = 'store_true', help = 'Use this flag if you only want to identify trays')
depthParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you need to rerun everything from the start (Otherwise only analysis files are recreated')

videoParser = subparsers.add_parser('VideoAnalysis', help='This command runs video analysis for a project: HMM->IdentifyClusters->SummarizeClusters->CreateVideoClips. Default is to use existing data except for the final step of creating clips.')
videoParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each video that should be analyzed')
videoParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
videoParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you need to rerun from the start')
videoParser.add_argument('-c', '--RewriteClusters', action = 'store_true', help = 'Use this flag if you need to rerun the creation of clusters, summary of clusters, and creation of clips')
videoParser.add_argument('-s', '--RewriteClusterSummaries', action = 'store_true', help = 'Use this flag if you need to rerun the creation of summary of clusters and creation of clips')
videoParser.add_argument('-f', '--FixIssues', action = 'store_true', help = 'Use this flag if you want to fix issues with the MC6_5 and MC16_2 cluster files')

MlabelParser = subparsers.add_parser('ManuallyLabelVideos', help='This command allows a user to manually label videos')
MlabelParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
MlabelParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to label.')
MlabelParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you would like to redo the labeling of the videos')

MlabelParser = subparsers.add_parser('CountFish', help='This command allows a user to manually label videos')
MlabelParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
MlabelParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to label.')
MlabelParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you would like to redo the labeling of the videos')

predictParser = subparsers.add_parser('PredictLabels', help='This command using machine learning to predict labels for each cluster')
predictParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
predictParser.add_argument('ModelNames', type = str, nargs = '+', help = 'Machine Learning Models to use to predict the cluster labels')
predictParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to label.')
predictParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you would like to redo the labeling of the videos')

summarizeParser = subparsers.add_parser('SummarizeProjects', help='This command summarizes data for the entire project')
summarizeParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
summarizeParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to label.')

trainParser = subparsers.add_parser('CreateModel', help='This command trains the 3DCNN model on a GPU machine')
trainParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project. This file must include a column with the ModelName with all of the videos that should be included in the model.')
trainParser.add_argument('classIndFile', type = str, help = 'Name of class file that contains info on labels')
trainParser.add_argument('ModelNames', nargs = '+', type = str, help = 'Name of model that will be trained')

args = parser.parse_args()

if args.command is None:
    parser.print_help()

elif args.command == 'CollectData':
    from Modules.Tracking.CichlidTracker import CichlidTracker
    while True:
        tracker = CichlidTracker(rcloneRemote + ':' + cloudMasterDirectory)

elif args.command in ['DepthAnalysis', 'VideoAnalysis', 'ManuallyLabelVideos', 'CountFish', 'PredictLabels', 'CreateModel']:

    import pandas as pd
    from Modules.Analysis.DataAnalyzer import DataAnalyzer as DA
    from Modules.Analysis.LabelAnalyzer import LabelAnalyzer as LA
    from Modules.Analysis.MachineLabel import MachineLearningMaker as MLM

    if args.command != 'CreateModel':
        inputData = ProjectData(args.InputFile, args.ProjectIDs)
    else:
        inputData = ProjectData(args.InputFile, None, args.ModelNames)
    
    if args.command == 'DepthAnalysis':
        print('Will analyze the following projects:', file = sys.stderr)
        print(','.join(inputData.projects), file = sys.stderr)
        # Depth Analysis requires user input. First get user input for all projects then allow depth analysis to run in the background
        for projectID in inputData.projects:
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.prepareData()
                if args.InitialPrep:
                    da_obj.cleanup()
                
        if not args.InitialPrep:
            for projectID in inputData.projects:
                with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                    da_obj.processDepth()
                    da_obj.cleanup()

    elif args.command == 'VideoAnalysis':
        for projectID, videos in inputData.clusterData.items():
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                if args.FixIssues:
                    da_obj.fixIssues(videos, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory)
                else:
                    da_obj.processVideos(videos, args.RewriteClusters, args.RewriteClusterSummaries, Nvideos = inputData.nVideos[projectID])
                da_obj.cleanup()

    elif args.command == 'ManuallyLabelVideos':
        print(inputData.manPredData)
        for projectID, videos in inputData.manPredData.items():
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.labelVideos(videos, manualLabelFile, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory)
                da_obj.cleanup()

    elif args.command == 'CountFish':
        print(inputData.manPredData)
        for projectID, videos in inputData.manPredData.items():
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.countFish(videos, rcloneRemote + ':' + cloudMasterDirectory + countingDirectory)
                #da_obj.cleanup()
                
    elif args.command == 'PredictLabels':
        if socket.gethostname() != 'biocomputesrg':
            raise Exception('PredictLabels analysis must be run on SRG or some other machine with good GPUs')
        print(os.environ['CONDA_DEFAULT_ENV'])
        for projectID, videos in inputData.clusterData.items():
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.predictLabels(videos, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory + 'Models/', args.ModelNames)
                
    elif args.command == 'CreateModel':
        if socket.gethostname() != 'biocomputesrg':
            raise Exception('TrainModel analysis must be run on SRG or some other machine with good GPUs')
        #if os.environ['CUDA_VISIBLE_DEVICES'] != '6':
        #    raise Exception('CUDA_VISIBLE_DEVICES is not set. Run "export CUDA_VISIBLE_DEVICES=6" and rerun')
        #print(os.environ['CONDA_DEFAULT_ENV'])
        print(inputData.mLearningData)
        processes = []
        confusionMatrices = []
        count = 0
        for mlID in inputData.mLearningData:
            ml_obj = MLM(mlID, inputData.mLearningData[mlID], localMasterDirectory + machineLearningDirectory, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory + 'Clips/', manualLabelFile, args.classIndFile)
            ml_obj.prepareData()
            processes.append(ml_obj.runTraining(GPU = count))
            count += 1

        for process in processes:
            process.communicate()

        for mlID in inputData.mLearningData:
            subprocess.Popen(['rclone', 'copy', localMasterDirectory + machineLearningDirectory + mlID, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory + mlID])

        #for projectID, videos in inputData.clusterData.items():
        #    with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
        #        da_obj.predictLabels(videos[projectID], rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory + '/Models/' + args.ModelName + '/')
                
elif args.command == 'SummarizeProjects':
    pass
        

