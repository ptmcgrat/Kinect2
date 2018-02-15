import os, sys, io
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from seaborn import heatmap


#add delta value for frame and background
#make masterstart return 2 lines

class LogFormatError(Exception):
    pass

class FrameObj:
    def __init__(self, npy_file, pic_file, time, med, std, gp):
        self.npy_file = npy_file
        self.pic_file = pic_file
        self.time = time
        self.med = med
        self.std = std
        self.gp = gp

class LogParser:    
    def __init__(self, logfile):
        self.logfile = logfile
        self.master_directory = logfile.replace(logfile.split('/')[-1], '')
        self.depth_df = self.master_directory + 'Depth.npy'

    def parse_log(self):

        self.speeds = []
        self.frames = []
        self.backgrounds = []
        self.movies = []
        
        with open(self.logfile) as f:
            for line in f:
                info_type = line.split(':')[0]
                if info_type == 'MasterStart':
                    try:
                        self.system
                        self.device
                        self.camera
                        self.uname
                    except AttributeError:
                        self.system, self.device, self.camera, self.uname = self._ret_data(line, ['System', 'Device', 'Camera','Uname'])
                    else:
                        raise LogFormatError('It appears MasterStart is present twice in the Logfile. Unable to deal')

                if info_type == 'MasterRecordInitialStart':
                    self.master_start = self._ret_data(line, ['Time'])

                if info_type == 'ROI':
                    try:
                        self.bounding_pic
                        self.bounding_shape
                    except AttributeError:
                        self.bounding_pic, self.bounding_shape = self._ret_data(line, ['Image', 'Shape'])
                        self.width = self.bounding_shape[3]
                        self.height = self.bounding_shape[2]
                    else:
                        raise LogFormatError('It appears ROI is present twice in the Logfile. Unable to deal')
                    
                if info_type == 'DiagnoseSpeed':
                    self.speed.append(self._ret_data(line, 'Rate'))
                    
                if info_type == 'FrameCaptured':
                    self.frames.append(FrameObj(self._ret_data(line, ['NpyFile','PicFile','Time','AvgMed','AvgStd','GP'])))

                if info_type == 'BackgroundCaptured':
                    self.backgrounds(FrameObj(self._ret_data(line, ['NpyFile','PicFile','Time','AvgMed','AvgStd','GP'])))
                    
                #if info_type == 'PiCameraStarted':
                    
        self.frames.sort(key = lambda x: x.time)
        self.backgrounds.sort(key = lambda x: x.time)

    def day_summary(self, day, start=10, stop=11):
        day_frames = [x for x in self.frames if x.time.day - self.master_start.day + 1 == day]

        start = [x for x in day_frames if x.time.hour == start][0]
        stop = [x for x in day_frames if x.time.hour == stop][0]
        start_pic = cv2.imread(start.pic_file)
        end_pic = cv2.imread(stop.pic_file)
        start_depth = np.load(start.npy_file)
        end_depth = np.load(stop.npy_file)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(3,2,1)       
        ax2 = fig.add_subplot(3,2,2)
        ax3 = fig.add_subplot(3,2,3)
        ax4 = fig.add_subplot(3,2,4)
        ax5 = fig.add_subplot(3,2,5)
        ax6 = fig.add_subplot(3,2,6)

        ax1.imshow(start_pic)
        ax2.imshow(end_pic)
        heatmap(end_depth - start_depth, ax = ax3, cbar = True) #seaborn function
        ax4.plot([x.time for x in day_frames], [x.med for x in day_frames])
        ax5.plot([x.time for x in day_frames], [x.std for x in day_frames])
        ax6.plot([x.time for x in day_frames], [x.n for x in day_frames])
        buf = io.StringIO()
        plt.savefig(buf, format='png')
        return buf.getvalue()
        
    def create_npy_array(self):
        self.all_data = np.empty(shape = (len(self.frames), self.width, self.height))
        for i, npy_file in enumerate(self.frames):
            data = np.load(npy_file)
            self.all_data[i] = data

        np.save(self.depth_df, self.all_data)


    def _ret_data(self, line, data):
        out_data = []
        if type(data) != list:
            data = [data]
        for d in data:
            t_data = line.split(d + ': ')[1].split(',,')[0]
            # Is it a date?
            try:
                out_data.append(dt.strptime(t_data, '%Y-%m-%d %H:%M:%S.%f'))
                continue
            except ValueError:
                pass
            # Is it a tuple?
            try:
                out_data.append(tuple(t_data))
                continue
            except TypeError:
                pass
            # Is it an int?
            try:
                out_data.append(int(t_data))
                continue
            except TypeError:
                pass
            # Is it a float?
            try:
                out_data.append(float(t_data))
            except TypeError:
                # Keep it as a string
                out_data.append(t_data)
        return out_data
    
