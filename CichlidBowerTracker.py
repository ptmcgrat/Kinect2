import argparse, os, socket
from subprocess import call

class ProjectData:
    def __init__(self, excelFile, projects, machLearningID = None):
        excelProjects = set()
        if machLearningID is None:
            dt = pd.read_excel(excelFile, header = 1, converters={'ClusterAnalysis':str,'ManualPrediction':str})
        else:
            dt = pd.read_excel(excelFile, header = 1, converters={'ClusterAnalysis':str,'ManualPrediction':str, machLearningID:str})
        self.projects = {}
        self.clusterData = {}
        self.manPredData = {}
        self.mLearningData = []
        for row in dt.iterrows():
            if row[1].Include == True:

                projectID = row[1].ProjectID
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
                if machLearningID is not None:
                    try:
                        if str(row[1][machLearningID]).lower() == 'true':
                            self.mLearningData.append(projectID)
                    except KeyError:
                        raise KeyError(machLearningID + ' not a column in Excel data file')
        for projectID in projects:
            if projectID not in excelProjects:
                    print('Cant find projectID: ' + projectID)
                    print('Options are ' ','.join(projects))


rcloneRemote = 'cichlidVideo' #The name of the rclone remote that has access to the dropbox or other cloud account
cloudMasterDirectory = 'McGrath/Apps/CichlidPiData/' # The master directory on the cloud account that stores video and depth data
localMasterDirectory = os.getenv('HOME') + '/Temp/CichlidAnalyzer/' #
machineLearningDirectory = '__MachineLearning/'
manualLabelFile = 'ManualLabeledClusters.csv' # The name of the file that contains info on all manually annotated data

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

trackerParser = subparsers.add_parser('CollectData', help='This command runs on Raspberry Pis to collect depth and RGB data')

depthParser = subparsers.add_parser('DepthAnalysis', help='This command runs depth analysis for a project')
depthParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
depthParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
depthParser.add_argument('-i', '--InitialPrep', action = 'store_true', help = 'Use this flag if you only want to identify trays and register RGB with depth data')
depthParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you need to rerun everything from the start (Otherwise only analysis files are recreated')

videoParser = subparsers.add_parser('VideoAnalysis', help='This command runs video analysis for a project: HMM->IdentifyClusters->SummarizeClusters->CreateVideoClips. Default is to use existing data except for the final step of creating clips.')
videoParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each video that should be analyzed')
videoParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
videoParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you need to rerun from the start')
videoParser.add_argument('-c', '--RewriteClusters', action = 'store_true', help = 'Use this flag if you need to rerun the creation of clusters, summary of clusters, and creation of clips')
videoParser.add_argument('-s', '--RewriteClusterSummaries', action = 'store_true', help = 'Use this flag if you need to rerun the creation of summary of clusters and creation of clips')

MlabelParser = subparsers.add_parser('ManuallyLabelVideos', help='This command allows a user to manually label videos')
MlabelParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
MlabelParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to label.')
MlabelParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you would like to redo the labeling of the videos')

predictParser = subparsers.add_parser('PredictLabels', help='This command using machine learning to predict labels for each cluster')
predictParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
predictParser.add_argument('ModelName', type = str, help = 'Machine Learning Model to use to predict the cluster labels')
predictParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to label.')
predictParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you would like to redo the labeling of the videos')

summarizeParser = subparsers.add_parser('SummarizeProjects', help='This command summarizes data for the entire project')
summarizeParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
summarizeParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to label.')

trainParser = subparsers.add_parser('CreateModel', help='This command trains the 3DCNN model on a GPU machine')
trainParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project. This file must include a column with the ModelName with all of the videos that should be included in the model.')
trainParser.add_argument('ModelName', type = str, help = 'Name of model that will be trained')
trainParser.add_argument('classIndFile', type = str, help = 'Name of class file that contains info on labels')

args = parser.parse_args()

if args.command is None:
    parser.print_help()

elif args.command == 'CollectData':
    from Modules.Tracking.CichlidTracker import CichlidTracker
    while True:
        tracker = CichlidTracker(rcloneRemote + ':' + cloudMasterDirectory)

elif args.command in ['DepthAnalysis', 'VideoAnalysis', 'ManuallyLabelVideos', 'PredictLabels', 'CreateModel']:

    import pandas as pd
    from Modules.Analysis.DataAnalyzer import DataAnalyzer as DA
    from Modules.Analysis.LabelAnalyzer import LabelAnalyzer as LA
    from Modules.Analysis.MachineLabel import MachineLabelCreator as MLC

    if args.command != 'CreateModel':
        inputData = ProjectData(args.InputFile, args.ProjectIDs)
    else:
        inputData = ProjectData(args.InputFile, None, args.ModelName)
    
    if args.command == 'DepthAnalysis':
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
                da_obj.processVideos(videos, args.RewriteClusters, args.RewriteClusterSummaries)
                da_obj.cleanup()

    elif args.command == 'ManuallyLabelVideos':
        print(inputData.manPredData)
        for projectID, videos in inputData.manPredData.items():
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.labelVideos(videos, manualLabelFile, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory)
                da_obj.cleanup()
                
    elif args.command == 'PredictLabels':
        if socket.gethostname() != 'biocomputesrg':
            raise Exception('PredictLabels analysis must be run on SRG or some other machine with good GPUs')
        print(os.environ['CONDA_DEFAULT_ENV'])
        for projectID, videos in inputData.clusterData.items():
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.predictLabels(videos[projectID], rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory + '/Models/' + args.ModelName + '/')
                
    elif args.command == 'CreateModel':
        #if socket.gethostname() != 'biocomputesrg':
        #    raise Exception('TrainModel analysis must be run on SRG or some other machine with good GPUs')
        #print(os.environ['CONDA_DEFAULT_ENV'])
        print(inputData.mLearningData)
        ml_obj = MLC(args.ModelName, inputData.mLearningData, localMasterDirectory + machineLearningDirectory, rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory, manualLabelFile, args.classIndFile)
        ml_obj.prepareData()
        #for projectID, videos in inputData.clusterData.items():
        #    with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
        #        da_obj.predictLabels(videos[projectID], rcloneRemote + ':' + cloudMasterDirectory + machineLearningDirectory + '/Models/' + args.ModelName + '/')
                
elif args.command == 'SummarizeProjects':
    pass
        

