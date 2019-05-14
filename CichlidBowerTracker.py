import argparse, os, socket
from subprocess import call


rcloneRemote = 'cichlidVideo' #The name of the rclone remote that has access to the dropbox or other cloud account
cloudMasterDirectory = 'McGrath/Apps/CichlidPiData/' # The master directory on the cloud account that stores video and depth data
localMasterDirectory = os.getenv('HOME') + '/Temp/CichlidAnalyzer/' # 

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

trackerParser = subparsers.add_parser('CollectData', help='This command runs on Raspberry Pis to collect depth and RGB data')

depthParser = subparsers.add_parser('DepthAnalysis', help='This command runs depth analysis for a project')
depthParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each project')
depthParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
depthParser.add_argument('-i', '--InitialPrep', action = 'store_true', help = 'Use this flag if you only want to identify trays and register RGB with depth data')
depthParser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you need to rerun everything from the start (Otherwise only analysis files are recreated')

videoParser = subparsers.add_parser('VideoAnalysis', help='This command runs video analysis for a project')
videoParser.add_argument('InputFile', type = str, help = 'Excel file containing information on each video that should be analyzed')
videoParser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
videoParser.add_argument('-a', '--Rewrite', action = 'store_true', help = 'Use this flag if you need to rerun from the start')

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

trainParser = subparsers.add_parser('TrainModel', help='This command trains the 3DCNN model on a GPU machine')

args = parser.parse_args()

if args.command is None:
    parser.print_help()

elif args.command == 'CollectData':
    from Modules.Tracking.CichlidTracker import CichlidTracker
    while True:
        tracker = CichlidTracker(rcloneRemote + ':' + cloudMasterDirectory)

elif args.command in ['DepthAnalysis', 'VideoAnalysis', 'ManuallyLabelVideos', 'PredictLabels']:

    import pandas as pd
    from Modules.Analysis.DataAnalyzer import DataAnalyzer as DA
    from Modules.Analysis.LabelAnalyzer import LabelAnalyzer as LA

    projects = {}
    videos = {}
    dt = pd.read_excel(args.InputFile)

    for row in dt.iterrows():
        if row[1].Include == True:
            projectID = row[1].ProjectID
            groupID = row[1].GroupID
            try:
                video = [int(x) for x in str(row[1].Videos).split(',')]
            except ValueError:
                video = None
            if args.ProjectIDs is None or projectID in args.ProjectIDs:
                projects[projectID] = groupID
                videos[projectID] = video

    if args.ProjectIDs is not None:
        for projectID in args.ProjectIDs:
            if projectID not in projects:
                print(projectID + ' not found in ' + args.InputFile + ' and will not be analyzed')

    if args.command == 'DepthAnalysis':
        # Depth Analysis requires user input. First get user input for all projects then allow depth analysis to run in the background
        for projectID in projects:
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.prepareData()
                da_obj.cleanup()

        for projectID in projects:
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                if not args.InitialPrep:
                    da_obj.processDepth()
                da_obj.cleanup()

    elif args.command == 'VideoAnalysis':
        for projectID in projects:
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.processVideos(videos[projectID], args.Rewrite)
                da_obj.cleanup()

    elif args.command == 'ManuallyLabelVideos':
        for projectID in projects:
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.labelVideos(videos[projectID], 'ManualLabeledClusters.csv', rcloneRemote + ':' + cloudMasterDirectory + '__MachineLearning/')

    elif args.command == 'PredictLabels':
        if socket.gethostname() != 'biocomputesrg':
            print('PredictLabels analysis must be run on SRG or some other machine with good GPUs')
            raise Exception
        print(os.environ['CONDA_DEFAULT_ENV'])
        for projectID in projects:
            with DA(projectID, rcloneRemote, localMasterDirectory, cloudMasterDirectory, args.Rewrite) as da_obj:
                da_obj.predictLabels(videos[projectID], rcloneRemote + ':' + cloudMasterDirectory + '__MachineLearning/Models/' + args.ModelName + '/')
elif args.command == 'SummarizeProjects':
    pass
        
elif args.command == 'TrainModel':
    pass
