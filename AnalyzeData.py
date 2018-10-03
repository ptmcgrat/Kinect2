from Modules.DataAnalyzer import DataAnalyzer as DA
import argparse, os, pickle
import pandas as pd

rcloneName = 'cichlidVideo'
dBoxMasterDir = 'McGrath/Apps/CichlidPiData/'
locMasterDir = os.getenv('HOME') + '/Temp/CichlidAnalyzer/'

parser = argparse.ArgumentParser()
parser.add_argument('InputFile', type = str, help = 'Excel file containing information on what you want analyzed')
parser.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Filter the name of the projects you would like to analyze.')
parser.add_argument('-d', '--delete', action = 'store_true', help = 'Delete existing data and reanalyze.')
parser.add_argument('-k', '--keep', action = 'store_true', help = 'Use this flag if you do not want the temp data to be deleted')
parser.add_argument('-T', '--Tray', action = 'store_true', help = 'Use this flag if you  want to identify the trays for each project (Useful if you are not on a computer with a good internet connection or is slow')
parser.add_argument('-D', '--Depth', action = 'store_true', help = 'Use this flag if you  want to analyze the depth data')
parser.add_argument('-H', '--Histogram', action = 'store_true', help = 'Use this flag if you  want analyze the data histograms')
parser.add_argument('-V', '--Videos', action = 'store_true', help = 'Use this flag if you  want to analyze the video files for a project (Useful if you already analyzed the depth data')

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
            video = [0]
        if args.ProjectIDs is None or projectID in args.ProjectIDs:
            projects[projectID] = groupID
            videos[projectID] = video

if args.delete:
    delete = True
else:
    delete = False
            
if args.Tray:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, [0], delete) as da_obj:
            da_obj.prepareData()

if args.Depth:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, [0], delete) as da_obj:
            da_obj.prepareData()
            da_obj.processDepth()

if args.Videos:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, [0], delete) as da_obj:
            if videos[projectID] == [0]:
                da_obj.processVideo()
            else:
                for index in videos[projectID]:
                    da_obj.processVideo(index)

if args.Histogram:
    for projectID in projects:
        with DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, [0], delete) as da_obj:
            histograms[projectID] = da_obj.retHistogramData()

            with open('histograms.pickle', 'wb') as handle:
                pickle.dump(histograms, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    


