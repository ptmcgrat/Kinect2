import os, sys, io
import numpy as np
from datetime import datetime as dt


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
                    t_list = self._ret_data(line, ['NpyFile','PicFile','Time','AvgMed','AvgStd','GP'])
                    t_list[0] = self.master_directory + t_list[0]
                    t_list[1] = self.master_directory + t_list[1]
                    self.frames.append(FrameObj(*t_list))

                if info_type == 'BackgroundCaptured':
                    t_list = self._ret_data(line, ['NpyFile','PicFile','Time','AvgMed','AvgStd','GP'])
                    t_list[0] = self.master_directory + t_list[0]
                    t_list[1] = self.master_directory + t_list[1]
                    self.backgrounds.append(FrameObj(*t_list))
                    
                if info_type == 'PiCameraStarted':
                    t_list = self._ret_data(line,['Time','File', 'FrameRate'])
                    t_list.append(self.master_directory)
                    self.movies.append(MovieObj(*t_list))
                    
        self.frames.sort(key = lambda x: x.time)
        self.backgrounds.sort(key = lambda x: x.time)
        self.lastBackgroundCounter = len(self.backgrounds)
        self.lastFrameCounter=len(self.frames)
        self.lastVideoCounter=len(self.movies)

    def create_npy_array(self):
        self.all_data = np.empty(shape = (len(self.frames), self.height, self.width))
        for i, frame in enumerate(self.frames):
            data = np.load(frame.npy_file)
            self.all_data[i] = data

        np.save(self.depth_df, self.all_data)

    def load_npy_array(self):
        try:
            self.all_data = np.load(self.depth_df)
        except:
            self.create_npy_array()
        
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
    def __init__(self, time, movie_file, framerate, master_directory):
        self.time = time
        self.h264_file = master_directory + movie_file
        self.mp4_file = master_directory + movie_file.replace('.h264', '') + '.mp4'
        self.framerate = framerate
        self.master_directory = master_directory
        self.hmm_file = self.master_directory + movie_file.split('/')[1].split('.')[0] + '.hmm'
