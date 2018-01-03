import argparse
from Modules.CichlidTracker import CichlidTracker

#Set up the options for the program (Don't change anything)
parser = argparse.ArgumentParser()

parser.add_argument('ProjectName', type = str, help = 'Name of the project you would like to collect')
parser.add_argument('Time', type = int, help = 'Days to record')
parser.add_argument('-o', '--OutputDirectory', type = str, help = 'Specify the output directory to save to')
parser.add_argument('-r', '--rewrite', help = 'Write over existing data in directory', action = 'store_true')

args = parser.parse_args()

tracker = CichlidTracker(args.ProjectName, args.OutputDirectory, args.rewrite)
tracker.capture_frames(total_time = args.Time * 24 * 60 * 60)

