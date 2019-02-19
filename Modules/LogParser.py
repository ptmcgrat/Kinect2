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
                        self.tankID
                        self.projectID
                    except AttributeError:
                        self.system, self.device, self.camera, self.uname, self.tankID, self.projectID = self._ret_data(line, ['System', 'Device', 'Camera','Uname', 'TankID', 'ProjectID'])
                    else:
                        raise LogFormatError('It appears MasterStart is present twice in the Logfile. Unable to deal')
                    #Check if error in Marks file (tankId and projectID are swapped)
                    if self.projectID[0:2] == 'Tk':
                        temp = self.projectID
                        self.projectID = self.tankID
                        self.tankID = temp

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
                    # Is this a Mark file?
                    try:
                        t_list[2].year
                    except AttributeError:
                        print(line)
                        print('-' + t_list[2] + '-')
                    if t_list[2].year == 1900:
                        # Get date from directory files are stored in
                        t_date = dt.strptime(t_list[0].split('/')[0], '%B-%d-%Y')
                        t_list[2] = t_list[2].replace(year = t_date.year, month = t_date.month, day = t_date.day)
                    self.frames.append(FrameObj(*t_list))

                if info_type == 'BackgroundCaptured':
                    t_list = self._ret_data(line, ['NpyFile','PicFile','Time','AvgMed','AvgStd','GP'])
                    self.backgrounds.append(FrameObj(*t_list))
                    
                if info_type == 'PiCameraStarted':
                    if 'VideoFile' in line:
                        #Patricks logfile
                        t_list = self._ret_data(line,['Time','VideoFile', 'PicFile', 'FrameRate'])
                    else:
                        #Marks logfile
                        t_list = self._ret_data(line,['Time','File'])
                        t_list.extend(['Unknown', 30])

                    self.movies.append(MovieObj(*t_list))

        self.frames.sort(key = lambda x: x.time)

        # Process frames into days
        rel_day = 0
        cur_day = 0
        self.days = {}
        for index,frame in enumerate(self.frames):
            if frame.time.day != cur_day:
                if rel_day != 0:
                    self.days[rel_day][1] = index
                rel_day += 1
                self.days[rel_day] = [index,0]
                frame.rel_day = rel_day
            
            cur_day = frame.time.day

        self.days[rel_day][1] = index + 1
        self.numDays = len(self.days)
            
        self.backgrounds.sort(key = lambda x: x.time)
        self.lastBackgroundCounter = len(self.backgrounds)
        self.lastFrameCounter=len(self.frames)
        self.lastVideoCounter=len(self.movies)
        
    def _ret_data(self, line, data):
        out_data = []
        if type(data) != list:
            data = [data]
        for d in data:
            try:
                t_data = line.split(d + ': ')[1].split(',,')[0]
            except IndexError:
                try:
                    t_data = line.split(d + ':')[1].split(',,')[0]
                except IndexError:
                    out_data.append('Error')
                    continue
            # Is it a date?
            try:
                out_data.append(dt.strptime(t_data, '%Y-%m-%d %H:%M:%S.%f'))
                continue
            except ValueError:
                pass
            try:
                out_data.append(dt.strptime(t_data, '%a %b %d %H:%M:%S %Y'))
                continue
            except ValueError:
                pass
            try:
                out_data.append(dt.strptime(t_data, '%H:%M:%S'))
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
        self.rel_day = 0
        self.frameDir = npy_file.replace(npy_file.split('/')[-1],'')


class MovieObj:
    def __init__(self, time, movie_file, pic_file, framerate):
        self.time = time
        if '.mp4' in movie_file:
            self.mp4_file = movie_file
            self.h264_file =  movie_file.replace('.mp4', '') + '.h264'
        else:
            self.h264_file =  movie_file
            self.mp4_file =  movie_file.replace('.h264', '') + '.mp4'
        self.pic_file =  pic_file
        self.framerate = framerate
        self.hmm_file = self.mp4_file.replace('.mp4', '.hmm')
        self.movieDir = movie_file.replace(movie_file.split('/')[-1],'')
