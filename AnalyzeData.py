from Modules.DataAnalyzer import DataAnalyzer as DA
import argparse, os

rcloneName = 'cichlidVideo'
dBoxMasterDir = 'McGrath/Apps/CichlidPiData/'
locMasterDir = os.getenv('HOME') + '/Temp/CichlidAnalyzer/'

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-f', '--InputFile', type = str, help = 'Excel file containing information on what you want analyzed')
group.add_argument('-p', '--ProjectIDs', nargs = '+', type = str, help = 'Name of the projects you would like to analyze.')
parser.add_argument('-v', '--Videos', nargs = '+', type = int, help = 'If you want to restrict video analysis to certain days use this flag (first day = 1, 0 = no video analysis, -1 = all)')
parser.add_argument('-k', '--keep', action = 'store_true', help = 'Use this flag if you do not want the temp data to be deleted')
args = parser.parse_args()

projects = {}
da_objs = {}
if args.InputFile is not None:
    if args.Videos is not None:
        print('Use of -f and -v option is not allowed. Ignoring info provided in -v', file = sys.stderr)
    #parse input file

else:
    if args.Videos is None:
        print('No -v argument provided. Will not analyze any videos')
        args.Videos = [0]
    if len(args.ProjectIDs) > 1 and args.Videos not in [[0],[-1]]:
        print('If more than one projectID is provided, -v flag can only be 0 (nothing analyzed) or -1 (everything analyzed). Defaulting to 0', file = sys.stderr)
        args.Videos = [0]
    for projectID in args.ProjectIDs:
        projects[projectID] = args.Videos

for projectID in projects:
    da_objs[projectID] = DA(projectID, rcloneName, locMasterDir, dBoxMasterDir, projects[projectID])
    da_objs[projectID].processDepth()

#for projectID, da_obj in da_objs.items():
#    da_obj.summarizeDepth()

