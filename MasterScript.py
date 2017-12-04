import argparse
from CichlidTracker import CichlidTracker

#Set up the options for the program (Don't change anything)
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

Collect_parser = subparsers.add_parser('CollectData', help='Collect ')
Collect_parser.add_argument('ProjectName', type = str, help = 'Name of the project you would like to collect')
Collect_parser.add_argument('Time', type = int, help = 'Days to record')
Collect_parser.add_argument('-o', '--OutputDirectory', type = str, help = 'Specify the output directory to save to')
Collect_parser.add_argument('-r', '--rewrite', help = 'Write over existing data in directory', action = 'store_true')

Analyze_parser = subparsers.add_parser('AnalyzeKinect2', help='Collect ')
Analyze_parser.add_argument('ProjectName', type = str, help = 'Name of the project you would like to collect')
Analyze_parser.add_argument('-o', '--odroid', help = 'This is running on an odroid', action = 'store_true')

args = parser.parse_args()
if args.command == 'CollectData':
    tracker = CichlidTracker(args.ProjectName, args.OutputDirectory, args.rewrite)
    tracker.capture_frames(total_time = args.Time * 24 * 60 * 60)

#if args.command == 'AnalyzeKinect2':
#    kt_obj = Kinect2Analyzer(args.ProjectName,args.odroid)
    #kt_obj.parse_log()
    #kt_obj.smooth_data()
    #kt_obj.select_regions()
    #kt_obj.create_heatmap_video()
#    kt_obj.summarize_data()
                
