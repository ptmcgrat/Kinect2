from Modules.VideoProcessor import VideoProcessor
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('VideoFiles', type = str, nargs = '+', help = 'Name of the videos you would like to analyze')
args = parser.parse_args()

for videofile in args.VideoFiles:
    obj = VideoProcessor(videofile)
    obj.summarize_data()
    continue
    try:
        obj = VideoProcessor(videofile)
    
        obj.convertVideo()
        obj.calculateHMM(delete = True)
        obj.summarize_data()
    except:
        print('Error in ' + videofile)
#    obj.summarize_data()
#results = pool.map(solve1, args)




#if args.command == 'AnalyzeKinect2':
#    kt_obj = Kinect2Analyzer(args.ProjectName,args.odroid)
    #kt_obj.parse_log()
    #kt_obj.smooth_data()
    #kt_obj.select_regions()
    #kt_obj.create_heatmap_video()
#    kt_obj.summarize_data()
                
