import os, sys, io
import numpy as np
from datetime import datetime as dt
import matplotlib
matplotlib.use('Pdf') # Enables creation of pdf without needing to worry about X11 forwarding when ssh'ing into the Pi
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from PIL import Image


#add delta value for frame and background
#make masterstart return 2 lines

class LogFormatError(Exception):
    pass

class LogParser:    
    def __init__(self, logfile):
        self.logfile = logfile
        self.master_directory = logfile.replace(logfile.split('/')[-1], '') + '/'
        self.depth_df = self.master_directory + 'MasterDepth.npy'
        self.parse_log()

    def parse_log(self):

        self.speeds = []
        self.frames = []
        self.backgrounds = []
        self.movies = []
        
        with open(self.logfile) as f:
            for line in f:
                line = line.rstrip()
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
                    self.master_start = self._ret_data(line, ['Time'])[0]

                if info_type == 'ROI':
                    try:
                        self.bounding_pic
                        self.bounding_shape
                    except AttributeError:
                        self.bounding_pic, self.bounding_shape = self._ret_data(line, ['Image', 'Shape'])
                        self.width = self.bounding_shape[2]
                        self.height = self.bounding_shape[3]
                    else:
                        raise LogFormatError('It appears ROI is present twice in the Logfile. Unable to deal')
                    
                if info_type == 'DiagnoseSpeed':
                    self.speeds.append(self._ret_data(line, 'Rate'))
                    
                if info_type == 'FrameCaptured':
                    self.frames.append(FrameObj(*self._ret_data(line, ['NpyFile','PicFile','Time','AvgMed','AvgStd','GP'])))

                if info_type == 'BackgroundCaptured':
                    self.backgrounds.append(FrameObj(*self._ret_data(line, ['NpyFile','PicFile','Time','AvgMed','AvgStd','GP'])))
                    
                if info_type == 'PiCameraStarted':
                    self.movies.append(MovieObj(*self._ret_data(line,['Time','File'])))
                    
        self.frames.sort(key = lambda x: x.time)
        self.backgrounds.sort(key = lambda x: x.time)
        self.lastBackgroundCounter = len(self.backgrounds)
        self.lastFrameCounter=len(self.frames)
        self.lastVideoCounter=len(self.movies)

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

    def drive_summary(self, fname):
        self.drive_summary_fname = fname
        last_hour_datetime = self.frames[-1].time - timedelta(hours = 1)  #uses last frame's time and subtracts one hour
        last_day_datetime = self.frames[-1].time - timedelta(days = 1)    #uses last frame's day and subtracts one day
        self.hour_frames = [w for w in x.frames if last_hour_datetime < w.time]  
        self.last_day_frames = [w for w in x.frames if last_day_datetime < w.time]
        
        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        ### TITLES ###
        ax1.set_title("Change over last hour")
        ax2.set_title("Change over last day")
        ax3.set_title("Change since beginning")
        ax4.set_title("Pixel quality since begining")
        ### AXES LABELS ###
        ax1.set_xlabel(self.hour_frames[0].time.strftime("%d %b %Y %H:%m") + " to " +
                       self.hour_frames[-1].time.strftime("%d %b %Y %H:%m"))
        ax1.set_ylabel("Med")
        ax2.set_xlabel(self.last_day_frames[0].time.strftime("%d %b %Y %H:%m") + " to " +
                       self.last_day_frames[-1].time.strftime("%d %b %Y %H:%m"))
        ax2.set_ylabel("Med")
        ax3.set_ylabel("Med")
        ax4.set_ylabel("% of GP")
        ### PLOT DATA ###
        ax1.plot([w.time for w in self.hour_frames], [w.med for w in self.hour_frames])  #change over last hour
        ax2.plot([w.time for w in self.last_day_frames], [w.med for w in self.last_day_frames])  #change over last hour
        ax3.plot([w.time for w in self.frames], [w.med for w in self.frames])   #change since beginning 
        ax4.plot([w.time for w in self.frames], [w.gp[0]/w.gp[1] for w in self.frames])  #pixel quality since beginning
        ### FORMAT X-AXIS ###
        ax1.xaxis_date()
        ax2.xaxis_date()
        ax3.xaxis_date()
        ax4.xaxis_date()
        
        hour_min_Fmt = mdates.DateFormatter("%H:%M")
        month_day_Fmt = mdates.DateFormatter("%d %b %Y")
        
        ax1.xaxis.set_major_formatter(hour_min_Fmt)
        ax2.xaxis.set_major_formatter(hour_min_Fmt)
        ax3.xaxis.set_major_formatter(month_day_Fmt)
        ax4.xaxis.set_major_formatter(month_day_Fmt)
        
        #fig.autofmt_xdate(rotation = 30, ha = 'right')  #doesn't work b/c will miss first row
        
        for label in ax1.get_xmajorticklabels()+ax2.get_xmajorticklabels()+ax3.get_xmajorticklabels()+ax4.get_xmajorticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
        
        plt.subplots_adjust(bottom = 0.15, left = 0.12, wspace = 0.24, hspace = 0.57)
        plt.savefig(self.drive_summary_fname)
        return self.drive_summary_fname
    
    def create_npy_array(self):
        self.all_data = np.empty(shape = (len(self.frames), self.height, self.width))
        for i, frame in enumerate(self.frames):
            data = np.load(self.master_directory + frame.npy_file)
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
            if t_data[0] == '(' and t_data[-1] == ')':
                out_data.append(tuple(int(x) for x in t_data[1:-1].split(', ')))
                continue
            # Is it an int?
            try:
                out_data.append(int(t_data))
                continue
            except ValueError:
                pass
            # Is it a float?
            try:
                out_data.append(float(t_data))
            except ValueError:
                # Keep it as a string
                out_data.append(t_data)
        return out_data
    

class FrameObj:
    def __init__(self, npy_file, pic_file, time, med, std, gp):
        self.npy_file = npy_file
        self.pic_file = pic_file
        self.time = time
        self.med = med
        self.std = std
        self.gp = gp

class MovieObj:
    def __init__(self, time, movie_file):
        self.time = time
        self.movie_file = movie_file
