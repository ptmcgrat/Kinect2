from Modules.DataAnalyzer import DataAnalyzer as DA
import argparse, os, pickle
import pandas as pd

rcloneName = 'cichlidVideo' #The name of the rclone remote that has access to the dropbox or other cloud account
dBoxMasterDir = 'McGrath/Apps/CichlidPiData/' # The master directory on the cloud account that stores video and depth data
locMasterDir = os.getenv('HOME') + '/Temp/CichlidAnalyzer/' # 

parser = argparse.ArgumentParser()
parser.add_argument('InputFile', type = str, help = 'Excel file containing information on what you want analyzed')
parser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
parser.add_argument('-r', '--Rewrite', action = 'store_true', help = 'Use this flag if you need to rerun from the start')
parser.add_argument('-T', '--Tray', action = 'store_true', help = 'Use this flag if you just  want to identify the trays for each project (Useful to identify all of the trays for a large number of projects before you run depth or video analysis')
parser.add_argument('-D', '--Depth', action = 'store_true', help = 'Use this flag if you  want to analyze the depth data')
parser.add_argument('-H', '--Histogram', action = 'store_true', help = 'Use this flag if you  want analyze the data histograms')
parser.add_argument('-V', '--Videos', action = 'store_true', help = 'Use this flag if you  want to analyze the video files for a project (Useful if you already analyzed the depth data')
parser.add_argument('-L', '--Label', action = 'store_true', help = 'Use this flag if you  want to label a subset of clusters for machine learning')

args = parser.parse_args()

projects = {}
histograms = {}
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
                        
if args.Tray:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, args.Rewrite) as da_obj:
            da_obj.identifyTray()

if args.Depth:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, args.Rewrite) as da_obj:
            da_obj.processDepth()

if args.Videos:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, args.Rewrite) as da_obj:
            print(videos[projectID])
            da_obj.processVideos(videos[projectID])
            da_obj.cleanup()
                
if args.Label:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, args.Rewrite) as da_obj:
            da_obj.labelVideos(videos[projectID])
      
if args.Histogram:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, args.Rewrite) as da_obj:
            histograms[projectID] = da_obj.retHistogramData()

            with open('histograms.pickle', 'wb') as handle:
                pickle.dump(histograms, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    


