import matplotlib as mpl
mpl.use('TkAgg')
import os, cv2, sys, datetime, shutil, pims, seaborn
from skimage import morphology
from random import randint
import scipy.signal
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as img
from matplotlib import animation
import Modules.LogParser as LP
import subprocess
from Modules.VideoProcessor import VideoProcessor
from Modules.roipoly import roipoly

class DataAnalyzer:
    def __init__(self, projectID, remote, locDir, cloudDir, videos, rewriteFlag = False):
        
        # Store input
        self.projectID = projectID
        self.remote = remote
        self.locMasterDir = locDir + projectID + '/'
        self.locAnalysisDir = self.locMasterDir + 'SubAnalysis/'
        self.remMasterDir = cloudDir + projectID + '/'
        self.remAnalysisDir = self.remMasterDir + 'SubAnalysis/'
        self.locOutputDir = self.locMasterDir + 'Output/'
        self.remOutputDir = self.remMasterDir + 'Output/'
                
        self.rewriteFlag = rewriteFlag
        
        # Delete and create local directories as necessary
        if self.rewriteFlag:
            try:
                shutil.rmtree(self.locMasterDir)
            except FileNotFoundError:
                pass

        if not os.path.exists(self.locAnalysisDir):
            os.makedirs(self.locAnalysisDir)

        if not os.path.exists(self.locOutputDir):
            os.makedirs(self.locOutputDir)

        self.anLF = open(self.locMasterDir + 'SubAnalysis/AnalysisLogfile.txt', 'a+')
        self.startTime = datetime.datetime.now()

            
        # Download, parse logfile, and verify projectIDs match
        subprocess.call(['rclone', 'copy', remote + ':' + self.remMasterDir + 'Logfile.txt', self.locMasterDir])
        self.lp = LP.LogParser(self.locMasterDir + 'Logfile.txt')
        if self.lp.projectID != self.projectID:
            self._print('ProjectID from logfile: ' + self.lp.projectID + 'does not match projectID folder: ' + self.projectID)
            sys.exit()

        if videos == [0]:
            self.videos = []
        elif videos == [-1]:
            self.videos = [self.lp.movies]
        else:
            self.videos = [self.lp.movies[int(x) - 1] for x in videos]
            
        #Print out some useful info to the user
        self._print('Beginning analysis of: ' + self.projectID + ' taken from tank: ' + self.lp.tankID)
        self._print(str(len(self.lp.frames)) + ' total frames for this project from ' + str(self.lp.frames[0].time) + ' to ' + str(self.lp.frames[-1].time))
        self._print(str(len(self.lp.movies)) + ' total videos for this project, will analyze ' + str(len(self.videos)) + ' of them.')
        if self.rewriteFlag:
            self._print('All data will be reanalyzed from start to finish')

        # Name of analysis files that will be created
        #self.rawDepthDataFile = self.locAnalysisDir + 'rawDepthData.npy'
        #self.interpDepthDataFile = self.locAnalysisDir + 'interpDepthData.npy'
        self.interpDepthFile = 'interpDepthData.npy'
        self.smoothDepthFile = 'smoothedDepthData.npy'
        self.trayFile = 'TrayInfo.txt'
        self.bowerFile = 'BowerInfo.txt'
        self.transMFile = 'TransformationMatrix.npy'
        self.histogramFile = 'DataHistograms.xlsx'
        self.totalChangeFile = 'TotalChange.npy'
        self.dailyChangeFile = 'DailyChange.npy'
        self.dailyMaskFile = 'DailyMask.npy'
        self.hourlyChangeFile = 'HourlyChange.npy'
        self.overnightChangeFile = 'OvernightChange.npy'
        self.analysisVideo = 'InitialVideo.mp4'
        
        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')
        
        # Prepare data - these three functions require user input and should be run first
        #self._identifyTray()
        #self._identifyBower()
        #self._registerImages()        

    def __del__(self):
        # Remove local files once object is destroyed
        shutil.rmtree(self.locMasterDir)
        print('Deleting')
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False
        
    def prepareData(self):
        self._identifyTray()
        
    def processDepth(self):
        self._createSmoothDepthArray()
        self._createHistogramData()
        self._calculateBowerData()
        self._dailySummary(self.interpDepthData, 'Smooth')
        self._createInitialVideo()
        #plt.clf()
        #plt.plot(c1)
        #plt.plot(c2)
        #plt.plot(c3)
        #plt.savefig(self.smoothedHourly)

    def retHistogramData(self):
        histogramFile = self.locOutputDir + 'DataHistograms.xlsx'
        if os.path.isfile(histogramFile):
            dt = pd.read_excel(histogramFile)
            return dt
        else:
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remOutputDir + 'DataHistograms.xlsx', self.locOutputDir], stderr = self.fnull)
            if os.path.isfile(histogramFile):
                dt = pd.read_excel(histogramFile)
                return dt
            else:
                self._createHistogramData()
                dt = pd.read_excel(histogramFile)
                return dt
            
        raise FileNotFoundError('Cannot find DataHistogram file')
        
    def processVideo(self, index = None):
        if index is None:
            vos = self.lp.movies
        else:
            vos = [self.lp.movies[index]]
        for vo in vos:
            if not os.path.isfile(self.locMasterDir + vo.mp4_file):
                self._print(vo.mp4_file + ' not present in local path. Trying to find it remotely')
                subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + vo.mp4_file, self.locMasterDir + vo.movieDir], stderr = self.fnull)
            if not os.path.isfile(self.locMasterDir + vo.mp4_file):
                self._print(vo.mp4_file + ' not present in remote path. Trying to find h264 file and convert it to mp4')

                if not os.path.isfile(self.locMasterDir + vo.h264_file):
                    subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + vo.h264_file, self.locMasterDir + vo.movieDir], stderr = self.fnull)
                    if not os.path.isfile(self.locMasterDir + vo.h264_file):
                        self._print('Unable to find video file for vo.mp4_file. Unable to analyze')
                        return
                    
                subprocess.call(['ffmpeg', '-framerate', str(vo.framerate),'-i', self.locMasterDir + vo.h264_file, '-c:v', 'copy', self.locMasterDir + vo.mp4_file])
                
                if os.stat(self.locMasterDir + vo.mp4_file).st_size >= os.stat(self.locMasterDir + vo.h264_file).st_size:
                    try:
                        vid = pims.Video(self.locMasterDir + vo.mp4_file)
                        vid.close()
                        os.remove(self.locMasterDir + vo.h264_file)
                    except Exception as e:
                        self._print(e)
                        continue
                    
                subprocess.call(['rclone', 'copy', self.locMasterDir + vo.mp4_file, self.remote + ':' + self.remMasterDir + vo.movieDir], stderr = self.fnull)

            baseName = vo.mp4_file.split('/')[-1].split('.')[0]
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + baseName, self.locAnalysisDir + baseName], stderr = self.fnull)
            self.vp_obj = VideoProcessor(self.locMasterDir + vo.mp4_file, self.locAnalysisDir + baseName)
            self.vp_obj.calculateHMM()
            self.vp_obj.createFramesToAnnotate()
            self.vp_obj.clusterHMM()
            subprocess.call(['rclone', 'copy', self.locAnalysisDir + baseName, self.remote + ':' + self.remAnalysisDir + baseName], stderr = self.fnull)
            shutil.rmtree(self.locAnalysisDir + basename)


    def _identifyTray(self):

        # If tray attribute already exists, exit
        try:
            self.tray_r
            return
        except AttributeError:
            pass

        # First deterimine if we want to recreate the tray region or try to use what already exists
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.trayFile, self.locAnalysisDir], stderr = self.fnull)

        if os.path.isfile(self.locAnalysisDir + self.trayFile):
            self._print('Loading tray information from file on dropbox')
            with open(self.locAnalysisDir + self.trayFile) as f:
                line = next(f)
                tray = line.rstrip().split(',')
                self.tray_r = [int(x) for x in tray]
            return

        # Unable to load it from existing file, either because it doesn't exist or the 
        self._print('Identify the parts of the frame that include tray to analyze')
      
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[0].npy_file, self.locMasterDir + self.lp.frames[0].frameDir], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[-1].npy_file, self.locMasterDir + self.lp.frames[-1].frameDir], stderr = self.fnull)
        if not os.path.isfile(self.locMasterDir + self.lp.frames[0].npy_file) or not os.path.isfile(self.locMasterDir + self.lp.frames[-1].npy_file):
            self._print('Cant find files needed to find the tray! Quitting')
            sys.exit()
        cmap = plt.get_cmap('jet')
        final_image = cmap(plt.Normalize(-10,10)(np.load(self.locMasterDir + self.lp.frames[-1].npy_file) -  np.load(self.locMasterDir + self.lp.frames[0].npy_file)))

        cv2.imshow('Identify the parts of the frame that include tray to analyze', final_image)
        tray_r = cv2.selectROI('Identify the parts of the frame that include tray to analyze', final_image, fromCenter = False)
        tray_r = tuple([int(x) for x in tray_r]) # sometimes a float is returned
        self.tray_r = [tray_r[1],tray_r[0],tray_r[1] + tray_r[3], tray_r[0] + tray_r[2]] # (x0,y0,xf,yf)
        # if bounding box is close to edge just set it as the edge
        if self.tray_r[0] < 50: 
            self.tray_r[0] = 0
        if self.tray_r[1] < 50: 
            self.tray_r[1] = 0
        if final_image.shape[0] - self.tray_r[2]  < 50: 
            self.tray_r[2] = final_image.shape[0]
        if final_image.shape[1] - self.tray_r[3]  < 50:  
            self.tray_r[3] = final_image.shape[1]
                
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        with open(self.locAnalysisDir + self.trayFile, 'w') as f:
            print(','.join([str(x) for x in self.tray_r]), file = f)

        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.trayFile, self.remote + ':' + self.remAnalysisDir])

    def _registerImages(self):

        # If tansformation matrix attribute already exists, exist
        try:
            self.transM
            return
        except AttributeError:
            pass

       # First deterimine if we want to recreate the tray region or try to use what already exists
        if not self.rewriteFlag:
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.transMFile, self.locAnalysisDir], stderr = self.fnull)

            if os.path.isfile(self.locAnalysisDir + self.transMFile):
                self._print('Loading transformation matrix information from file on dropbox')
                self.transM = np.load(self.locAnalysisDir + self.transMFile)
                return
                
        # Unable to load it from existing file, either because it doesn't exist or the 
        self._print('Trying to register RGB and Depth data ')
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[0].pic_file, self.locMasterDir + self.lp.frames[0].frameDir], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.movies[0].pic_file, self.locMasterDir + self.lp.movies[0].frameDir], stderr = self.fnull)
        if not os.path.isfile(self.locMasterDir + self.lp.frames[0].pic_file) or not os.path.isfile(self.locMasterDir + self.lp.movies[0].pic_file):
            self._print('Cant find RGB pictures of both ')
            self.transM = None
            return

        im1 =  cv2.imread(self.locMasterDir + self.lp.frames[0].pic_file)
        im2 =  cv2.imread(self.locMasterDir + self.lp.movies[0].pic_file)

        im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)       
        ax2 = fig.add_subplot(1,2,2)
        
        ax1.imshow(im1_gray, cmap='gray')
        ax2.imshow(im2_gray, cmap='gray')

        ax1.set_title('Select four points in this object (Control-click on the fifth point exits)')
        ROI1 = roipoly(roicolor='r')
        plt.show()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)       
        ax2 = fig.add_subplot(1,2,2)
  
        ax1.imshow(im1_gray, cmap='gray')
        ROI1.displayROI(ax = ax1)
        ax2.imshow(im2_gray, cmap='gray')

        ax2.set_title('Select four points in this object (Control-click on the fifth point exits)')
        ROI2 = roipoly(roicolor='b')
        plt.show()

        ref_points =[[ROI1.allxpoints[0], ROI1.allypoints[0]], [ROI1.allxpoints[1], ROI1.allypoints[1]], [ROI1.allxpoints[2], ROI1.allypoints[2]], [ROI1.allxpoints[3], ROI1.allypoints[3]]]
        new_points =[[ROI2.allxpoints[0], ROI2.allypoints[0]], [ROI2.allxpoints[1], ROI2.allypoints[1]], [ROI2.allxpoints[2], ROI2.allypoints[2]], [ROI2.allxpoints[3], ROI2.allypoints[3]]]
        self.transM = cv2.getPerspectiveTransform(np.float32(ref_points),np.float32(new_points))
        np.save(self.locAnalysisDir + self.transMFile, self.transM)

        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.transMFile, self.remote + ':' + self.remAnalysisDir], stderr = self.fnull)

        #self.pit_mask_PiCamera = cv2.warpPerspective(self.pit_mask, self.M, (1296,972))
        #self.castle_mask_PiCamera = cv2.warpPerspective(self.castle_mask, self.M, (1296,972))
                
    def _createSmoothDepthArray(self, totalGoodData = 0.3, minGoodData = 0.5, minUnits = 5, tunits = 71, order = 4):
        #Is it already loaded?
        try:
            self.smoothDepthData
            return
        except AttributeError:
            pass

        # First deterimine if we want to recreate the tray region or try to use what already exists
        if not self.rewriteFlag:
            if not os.path.isfile(self.locAnalysisDir + self.smoothDepthFile):
                subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.smoothDepthFile, self.locAnalysisDir], stderr = self.fnull)
                
            if not os.path.isfile(self.locAnalysisDir + self.interpDepthFile):
                subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.interpDepthFile, self.locAnalysisDir], stderr = self.fnull)

                
            if os.path.isfile(self.locAnalysisDir + self.smoothDepthFile) and os.path.isfile(self.locAnalysisDir + self.interpDepthFile):
                print('Loading depth information from file on dropbox', file = sys.stderr)
                self.smoothDepthData = np.load(self.locAnalysisDir + self.smoothDepthFile)
                self.interpDepthData = np.load(self.locAnalysisDir + self.interpDepthFile)
                return

            
        #Load or calculate raw data
        print('Creating large npy array to hold raw depth data', file = sys.stderr)
        rawDepthData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
        frameDirectories = set()
        for i, frame in enumerate(self.lp.frames):
            if frame.frameDir not in frameDirectories:
                print('Downloading ' + frame.frameDir + ' from remote host', file = sys.stderr)
                subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + frame.frameDir, self.locMasterDir + frame.frameDir, '--exclude', '*.mp4', '--exclude', '*.h264'])
                print('Done', file = sys.stderr)
                frameDirectories.add(frame.frameDir)
                
            try:
                data = np.load(self.locMasterDir + frame.npy_file)
            except ValueError:
                print('Bad frame: ' + str(i) + ', ' + frame.npy_file, file = sys.stderr)
                rawDepthData[i] = self.rawDepthData[i-1]
            else:
                rawDepthData[i] = data

        # Convert to cm
        rawDepthData = 100/(-0.0037*rawDepthData + 3.33)
        rawDepthData[(rawDepthData < 40) | (rawDepthData > 80)] = np.nan
            
        # Make copy of raw data
        interpDepthData = rawDepthData.copy()

        # Count number of good pixels
        goodDataAll = np.count_nonzero(~np.isnan(interpDepthData), axis = 0)
        goodDataStart = np.count_nonzero(~np.isnan(interpDepthData[:100]), axis = 0)

        numFrames = len(self.lp.frames)
        nans = np.cumsum(np.isnan(interpDepthData), axis = 0)
        
        # Process each pixel
        print('Interpolating raw values', file = sys.stderr)
        for i in range(rawDepthData.shape[1]):
            if i % 250 == 0:
                print('Interpolating: Currently on row: ' + str(i))
            for j in range(rawDepthData.shape[2]):
                if goodDataAll[i,j] > totalGoodData*numFrames or goodDataStart[i,j] > minGoodData*100:
                    bad_indices = np.where(nans[minUnits:,i,j] - nans[:-1*minUnits,i,j] == minUnits -1)[0] + int(minUnits/2)+1
                    interpDepthData[bad_indices,i,j] = np.nan

                    nan_ind = np.isnan(interpDepthData[:,i,j])
                    x_interp = np.where(nan_ind)[0]
                    x_good = np.where(~nan_ind)[0]

                    l_data = interpDepthData[x_good[:10], i, j].mean()
                    r_data = interpDepthData[x_good[-10:], i, j].mean()

                    try:
                        interpDepthData[x_interp, i, j] = np.interp(x_interp, x_good, interpDepthData[x_good, i, j], left = l_data, right = r_data)
                    except ValueError:
                        print(str(x_interp) + ' ' + str(x_good))
                else:
                    interpDepthData[:,i,j] = np.nan
                        
        #Set data outside of tray to np.nan
        interpDepthData[:,:self.tray_r[0],:] = np.nan
        interpDepthData[:,self.tray_r[2]:,:] = np.nan
        interpDepthData[:,:,:self.tray_r[1]] = np.nan
        interpDepthData[:,:,self.tray_r[3]:] = np.nan

        self.interpDepthData = interpDepthData
        np.save(self.locAnalysisDir + self.interpDepthFile, self.interpDepthData)
        self.smoothDepthData = scipy.signal.savgol_filter(interpDepthData, tunits, order, axis = 0, mode = 'mirror')
        np.save(self.locAnalysisDir + self.smoothDepthFile, self.smoothDepthData)
        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.interpDepthFile, self.remote + ':' + self.remAnalysisDir], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.smoothDepthFile, self.remote + ':' + self.remAnalysisDir], stderr = self.fnull)

    def _calculateBowerData(self, day_threshold = 0.4, min_pixels = 250, hourlyChange = 2):
        self.firstDay = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        self.dailyChange = np.empty(shape = (self.lp.numDays, self.lp.height, self.lp.width))
        self.hourlyChange = np.empty(shape = (int(self.lp.numDays*24/hourlyChange), self.lp.height, self.lp.width))
        self.overnightChange = np.empty(shape = (self.lp.numDays, self.lp.height, self.lp.width))
        
        self.mask = np.empty(shape = (self.lp.numDays, self.lp.height, self.lp.width), dtype = bool)
        
        #DailyChange        
        for i in range(self.lp.numDays):
            
            start = self.firstDay + datetime.timedelta(hours = 24*i)
            end = self.firstDay + datetime.timedelta(hours = 24*(i+1))
            try:
                first_index = max([False if x.time<=start else True for x in self.lp.frames].index(True) - 1, 0) #This ensures that we get overnight changes when kinect wasn't running
            except ValueError:
                first_index = 0
            try:
                last_index = [False if x.time<=end else True for x in self.lp.frames].index(True) - 1
            except ValueError:
                last_index = len(self.lp.frames) - 1 

            if last_index != first_index:
                mask = self.smoothDepthData[first_index] - self.smoothDepthData[last_index]
                
            mask = np.absolute(mask)
            mask[mask < day_threshold] = 0
            mask[np.isnan(mask)] = 0
            self.mask[i] = morphology.remove_small_objects(mask.astype(bool), min_pixels)

            self.dailyChange[i] = (self.interpDepthData[first_index] - self.interpDepthData[last_index])

            if i == self.lp.numDays - 1:
                self.overnightChange[i] = self.interpDepthData[first_index] - self.interpDepthData[first_index]
            else:
                self.overnightChange[i] = self.interpDepthData[last_index] - self.interpDepthData[last_index+1]
                print(str(self.lp.frames[last_index].time) + ' to ' + str(self.lp.frames[last_index+1].time))

            
        for i in range(self.lp.numDays):
            for j in range(int(24/hourlyChange)):
                start = self.firstDay + datetime.timedelta(hours = 24*i + j*hourlyChange)
                end = self.firstDay + datetime.timedelta(hours = 24*i + (j+1)*hourlyChange)
                try:
                    first_index = max([False if x.time<=start else True for x in self.lp.frames].index(True), 0) #This ensures that we get overnight changes when kinect wasn't running
                except ValueError:
                    if self.lp.frames[0].time > start:
                        first_index = 0
                    elif self.lp.frames[-1].time < start:
                        first_index = len(self.lp.frames) - 1
                    else:
                        print('Not sure what to make first_index. Quitting...')
                        sys.exit()
                try:
                    last_index = [False if x.time<=end else True for x in self.lp.frames].index(True) - 1

                except ValueError:
                    if self.lp.frames[0].time > end:
                        last_index = 0
                    elif self.lp.frames[-1].time < end:
                        last_index = len(self.lp.frames) - 1
                    else:
                        print('Not sure what to make first_index. Quitting...')
                        sys.exit()
                        
                if abs(last_index - first_index) <= 1:
                    last_index = first_index


                #Calculate largest changing values
                tchange = self.interpDepthData[first_index] - self.interpDepthData[last_index]

                self.hourlyChange[i*int(24/hourlyChange) + j] = tchange


        np.save(self.locAnalysisDir + self.totalChangeFile, self.smoothDepthData[0] - self.smoothDepthData[-1])
        np.save(self.locAnalysisDir + self.dailyChangeFile, self.dailyChange)
        np.save(self.locAnalysisDir + self.dailyMaskFile, self.mask)
        np.save(self.locAnalysisDir + self.hourlyChangeFile, self.hourlyChange)
        np.save(self.locAnalysisDir + self.overnightChangeFile, self.overnightChange)
        
        subprocess.call(['rclone', 'copy', self.locAnalysisDir, self.remote + ':' + self.remAnalysisDir], stderr = self.fnull)

    def _createInitialVideo(self):
        
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        fig.suptitle(self.projectID + ' ' + str(self.lp.frames[0].time))
        
        v_min = -2
        v_max = 2

        try:
            img_0 = img.imread(self.locMasterDir + self.lp.frames[0].pic_file)
            ax1.imshow(img_0)
        except:
            ax1.imshow(self.interpDepthData[0] - self.interpDepthData[0], vmin = v_min, vmax = v_max)

        hm_0 = np.load(self.locMasterDir + self.lp.frames[0].npy_file)
        hm_0 = 100/(-0.0037*hm_0 + 3.33)

        ax2.imshow(hm_0 - hm_0, vmin = v_min, vmax = v_max)

        hm_1 = self.interpDepthData[0] - self.interpDepthData[0]
        ax3.imshow(hm_1, vmin = v_min, vmax = v_max)

        hm_2 = self.smoothDepthData[0] - self.smoothDepthData[0]
        ax4.imshow(hm_2, vmin = v_min, vmax = v_max)
        
        writer = animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Patrick McGrath'), bitrate=1800)
        anim = animation.FuncAnimation(fig, self._animate, frames = self.interpDepthData.shape[0], fargs = (fig, ax1, ax2, ax3, ax4, hm_0, v_min, v_max), repeat = False)
        print(self.locOutputDir + self.analysisVideo)
        anim.save(self.locOutputDir + self.analysisVideo, writer = writer)        
        subprocess.call(['rclone', 'copy', self.locOutputDir + self.analysisVideo, self.remote + ':' + self.remOutputDir], stderr = self.fnull)

    def _animate(self, i, fig, ax1, ax2, ax3, ax4, hm_0_0, v_min, v_max):
        if i % 80 == 0:
            print(str(i) + ' of ' + str(self.interpDepthData.shape[0]))
        fig.clf()

        fig.suptitle(self.projectID + ' ' + str(self.lp.frames[i].time))

        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        try:
            img_0 = img.imread(self.locMasterDir + self.lp.frames[0].pic_file)
            ax1.imshow(img_0)
        except:
            ax1.imshow(self.interpDepthData[0] - self.interpDepthData[0], vmin = v_min, vmax = v_max)

        hm_0 = np.load(self.locMasterDir + self.lp.frames[i].npy_file)
        hm_0 = 100/(-0.0037*hm_0 + 3.33) - hm_0_0
        
        ax2.imshow(hm_0, vmin = v_min, vmax = v_max)

        hm_1 = self.interpDepthData[i] - self.interpDepthData[0]
        ax3.imshow(hm_1, vmin = v_min, vmax = v_max)

        hm_2 = self.smoothDepthData[i] - self.smoothDepthData[0]
        ax4.imshow(hm_2, vmin = v_min, vmax = v_max)

    def _createHistogramData(self):
        self._print('Creating histogram data file for ' + self.projectID)
        self._createSmoothDepthArray()
        
        bins = np.array(list(range(-100, 101, 1)))*.2
        dt = pd.DataFrame(index = bins[0:-1])

        #TotalChange
        total_change = self.smoothDepthData[0] - self.smoothDepthData[-1]
        a,b = np.histogram(total_change[~np.isnan(total_change)], bins = bins)
        dt['Total'] = pd.Series(a, index = bins[:-1])
        
        #DailyChange
        start_day = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        for i in range(self.lp.numDays):
            start = start_day + datetime.timedelta(hours = 24*i)
            end = start_day + datetime.timedelta(hours = 24*(i+1))
            try:
                first_index = max([False if x.time<=start else True for x in self.lp.frames].index(True) - 1, 0) #This ensures that we get overnight changes when kinect wasn't running
            except ValueError:
                first_index = 0
            try:
                last_index = [False if x.time<=end else True for x in self.lp.frames].index(True) - 1
            except ValueError:
                last_index = len(self.lp.frames) - 1 

            if last_index != first_index:
                total_change = self.smoothDepthData[first_index] - self.smoothDepthData[last_index]
                a,b = np.histogram(total_change[~np.isnan(total_change)], bins = bins)

                dt['Day' + str(i+1) + '_' + str(first_index) + 'to' + str(last_index)] = pd.Series(a, index = bins[:-1])

        #HourlyChange
        start_day = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        for i in range(self.lp.numDays*24):
            start = start_day + datetime.timedelta(hours = 1*i)
            end = start_day + datetime.timedelta(hours = 1*(i+1))
            try:
                first_index = [False if x.time<=start else True for x in self.lp.frames].index(True)
            except ValueError:
                first_index = len(self.lp.frames) - 1
            try:
                last_index = max([False if x.time<=end else True for x in self.lp.frames].index(True) - 1, 0)
            except ValueError:
                last_index = len(self.lp.frames) - 1

            if last_index != first_index:
                total_change = self.smoothDepthData[first_index] - self.smoothDepthData[last_index]
                a,b = np.histogram(total_change[~np.isnan(total_change)], bins = bins)

                dt['Hour' + str(i+1) + '_' + str(first_index) + 'to' + str(last_index)] = pd.Series(a, index = bins[:-1])

        writer = pd.ExcelWriter(self.locOutputDir + 'DataHistograms.xlsx')
        dt.to_excel(writer, 'Histogram')
        writer.save()
        
        subprocess.call(['rclone', 'copy', self.locOutputDir + 'DataHistograms.xlsx', self.remote + ':' + self.remOutputDir], stderr = self.fnull)

        
    def _dailySummary(self, data, baseFileName, hour_threshold = .5, day_threshold = .4):

        self._print('Creating Daily Summary pdf for ' + self.projectID)
        
        # Create summary figure
        fig = plt.figure(figsize = (11,8.5)) 
        fig.suptitle(self.projectID)
        
        grid = plt.GridSpec(8, self.lp.numDays*4, wspace=0.02, hspace=0.02)

#        rect = patches.Rectangle((self.bowerMask[1],self.bowerMask[0]),self.bowerMask[3] - self.bowerMask[1],self.bowerMask[2] - self.bowerMask[0],linewidth=1,edgecolor='r',facecolor='none')
 #       rect2 = patches.Rectangle((self.bowerMask[1],self.bowerMask[0]),self.bowerMask[3] - self.bowerMask[1],self.bowerMask[2] - self.bowerMask[0],linewidth=1,edgecolor='r',facecolor='none')
        
        pic_ax = fig.add_subplot(grid[0:2, 0:self.lp.numDays*2])
        ax = pic_ax.imshow(data[-1][self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]])
        #pic_ax.add_patch(rect)
        pic_ax.set_title('Final depth (cm)')
        pic_ax.set_xticklabels([])
        pic_ax.set_yticklabels([])

        plt.colorbar(ax, ax = pic_ax)
        pic_ax2 = fig.add_subplot(grid[0:2, self.lp.numDays*2:])
        ax2 = pic_ax2.imshow(-1*(data[-1] - data[0])[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]], vmin = -5, vmax = 5)
        #pic_ax2.add_patch(rect2)
        pic_ax2.set_title('Total depth change (cm)')
        pic_ax2.set_xticklabels([])
        pic_ax2.set_yticklabels([])
        plt.colorbar(ax2, ax = pic_ax2)

        current_t = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        bigPixels = []
        volumeChange = []

        #DailyChange        
        start_day = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        for i in range(self.lp.numDays):
            if i == 0:
                current_ax = [fig.add_subplot(grid[2, i*4:i*4+3])]
                current_ax2 = [fig.add_subplot(grid[3, i*4:i*4+3], sharex = current_ax[i])]
                current_ax3 = [fig.add_subplot(grid[4, i*4:i*4+3], sharex = current_ax[i])]
                current_ax4 = [fig.add_subplot(grid[5, i*4:i*4+3])]
            else:
                current_ax.append(fig.add_subplot(grid[2, i*4:i*4+3], sharey = current_ax[0]))
                current_ax2.append(fig.add_subplot(grid[3, i*4:i*4+3], sharex = current_ax[i], sharey = current_ax2[0]))
                current_ax3.append(fig.add_subplot(grid[4, i*4:i*4+3], sharex = current_ax[i], sharey = current_ax3[0]))
                current_ax4.append(fig.add_subplot(grid[5, i*4:i*4+3], sharey = current_ax4[0]))

            start = start_day + datetime.timedelta(hours = 24*i)
            end = start_day + datetime.timedelta(hours = 24*(i+1))

            current_ax[i].set_title(str(start.date()))
            try:
                last_index = max([False if x.time<=end else True for x in self.lp.frames].index(True) - 1, 0)
            except:
                last_index = -1

            current_ax[i].imshow((self.smoothDepthData[0] - self.smoothDepthData[last_index])[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]], vmin = -4*day_threshold, vmax = 4*day_threshold)
            current_ax2[i].imshow(self.dailyChange[i][self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]], vmin = -4*day_threshold, vmax = 4*day_threshold)
            current_ax3[i].imshow(self.mask[i][self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]])

            bigPixels.append(np.count_nonzero(self.mask[i]))
            volumeChange.append(np.nansum(np.absolute(self.dailyChange[i][self.mask[i]==True])))
            
            
            #Calculate largest changing values
            tdata = self.dailyChange[i].copy()
            tdata[self.mask[i] == False] = 0
            tdata = tdata[~np.isnan(tdata)]
            abs_data = np.absolute(tdata)
            x_values = [-1, -1.33333, -1.666666, -2, -2.33333, -2.6666666, -3]
            y_values = []
            for power in x_values:
                try:
                    new_thresh = np.percentile(abs_data, (1-10**power)*100)
                except IndexError:
                    print("IndexError in threshold: " + str(power))
                    y_values.append(0)
                else:
                    y_values.append(np.nansum(tdata[abs_data > new_thresh]))
                
            current_ax4[i].plot(x_values, y_values)
            
            current_ax[i].set_xticklabels([])
            current_ax2[i].set_xticklabels([])
            current_ax3[i].set_xticklabels([])

            if i != 0:
                current_ax4[i].set_yticklabels([])

            current_ax[i].set_yticklabels([])
            current_ax2[i].set_yticklabels([])
            current_ax3[i].set_yticklabels([])
            current_ax[i].set_adjustable('box-forced')
            current_ax2[i].set_adjustable('box-forced')
            current_ax3[i].set_adjustable('box-forced')

        current_ax1 = fig.add_subplot(grid[6,0:self.lp.numDays*2])
        current_ax2 = fig.add_subplot(grid[6, self.lp.numDays*2:])

        current_ax1.plot(bigPixels)
        current_ax2.plot(volumeChange)
        
        plt.tight_layout()
        #plt.show()
            
        plt.savefig(self.locOutputDir + 'DailySummary.pdf')
        plt.clf()

        subprocess.call(['rclone', 'copy', self.locOutputDir + 'DailySummary.pdf', self.remote + ':' + self.remOutputDir], stderr = self.fnull)
    
        fig = plt.figure(figsize = (11,8.5)) 
        fig.suptitle(self.projectID)
        
        grid = plt.GridSpec(self.lp.numDays, 14, wspace=0.02, hspace=0.02)

        changes = []
        for i in range(0, self.lp.numDays):
            for j in range(12):
                current_ax = fig.add_subplot(grid[i, j])

                current_ax.imshow(self.hourlyChange[i*12 + j][self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]], vmin = -0.4, vmax = .4)
                current_ax.set_adjustable('box-forced')
                current_ax.set_xticklabels([])
                current_ax.set_yticklabels([])
                if i == 0:
                    current_ax.set_title(str(j*2) + '-' + str((j+1)*2))

            current_ax = fig.add_subplot(grid[i, 12])
            current_ax.imshow(self.mask[i][self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]])
            current_ax.set_adjustable('box-forced')
            current_ax.set_xticklabels([])
            current_ax.set_yticklabels([])
            if i==0:
                current_ax.set_title('Mask')


            current_ax = fig.add_subplot(grid[i, 13])
            current_ax.imshow(self.overnightChange[i][self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]], vmin = -0.4, vmax = .4)
            current_ax.set_adjustable('box-forced')
            current_ax.set_xticklabels([])
            current_ax.set_yticklabels([])
            if i==0:
                current_ax.set_title('O/N')


            
        plt.savefig(self.locOutputDir + 'HourlySummary.pdf')
        subprocess.call(['rclone', 'copy', self.locOutputDir + 'HourlySummary.pdf', self.remote + ':' + self.remOutputDir], stderr = self.fnull)

        plt.clf()

    def _print(self, outtext):
        now = datetime.datetime.now()
        print(str(now) + ': ' + outtext, file = self.anLF)
        print(outtext, file = sys.stderr)
