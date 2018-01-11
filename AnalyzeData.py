from Modules.VideoProcessor import VideoProcessor
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Modules.roipoly import roipoly
import pylab as pl


parser = argparse.ArgumentParser()
parser.add_argument('VideoFiles', type = str, nargs = '+', help = 'Name of the videos you would like to analyze')
args = parser.parse_args()

#for videofile in args.VideoFiles:
#    try:
#        obj = VideoProcessor(videofile)
#    
#        obj.convertVideo()
#        obj.calculateHMM(delete = True)
#        obj.summarize_data()
#    except:
#        print('Error in ' + videofile)
#    obj.summarize_data()
#results = pool.map(solve1, args)




#if args.command == 'AnalyzeKinect2':
#    kt_obj = Kinect2Analyzer(args.ProjectName,args.odroid)
    #kt_obj.parse_log()
    #kt_obj.smooth_data()
    #kt_obj.select_regions()
    #kt_obj.create_heatmap_video()
#    kt_obj.summarize_data()
                
# This is a hack



for i,videofile in enumerate(args.VideoFiles):
    print(videofile)
    obj = VideoProcessor(videofile)
    if i == 0:
        data = np.zeros(shape = (12*len(args.VideoFiles), obj.height, obj.width), dtype = 'uint8')
    data[i*12:i*12 + 12] = obj.summarize_data()

background = np.sum(data, axis = 0)
pl.imshow(background)
pl.colorbar()
pl.title('Identify regions with sand')
ROI = roipoly(roicolor='r')
plt.show()

edges = ROI.getEdges(background)
max_c = data.max()*(1/4)

rows = []
for i,videofile in enumerate(args.VideoFiles):
    row = []
    for j in range(0,12, 2):
        square = np.sum(data[i*12+j:i*12+j+2], axis = 0)/max_c*255
        square[edges] = 255
        row.append(square.copy()[0:700,600:1200])
    day_change = np.sum(data[i*12:(i+1)*12], axis = 0)/max_c*255
    day_change[edges] = 255
    accum_change = np.sum(data[0:(i+1)*12], axis = 0)/max_c/4*255
    accum_change[edges] = 255
    rows.append(np.concatenate(row + [day_change[0:700,600:1200], accum_change[0:700,600:1200]], axis = 1))

plt.imshow(np.concatenate(rows, axis = 0))
plt.show()


