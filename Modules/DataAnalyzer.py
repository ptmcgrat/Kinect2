import subprocess, os, sys, datetime, shutil

from Modules.DepthProcessor import DepthProcessor as DP
from Modules.LogParser import LogParser as LP
from Modules.VideoProcessor import VideoProcessor as VP

#import matplotlib as mpl
#mpl.use('TkAgg')
#import os, cv2, sys, datetime, shutil, pims, seaborn
#from skimage import morphology
#from random import randint
#import scipy.signal
#import numpy as np
#import pandas as pd
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import matplotlib.image as img
#from matplotlib import animation
#import subprocess
#from Modules.VideoProcessor import VideoProcessor
#from Modules.roipoly import roipoly

class DataAnalyzer:
    def __init__(self, projectID, remote, locDir, cloudDir, rewriteFlag):
        
        # Store input
        self.projectID = projectID
        self.remote = remote
        self.localMasterDirectory = locDir + projectID + '/'
        self.cloudMasterDirectory = remote + ':' + cloudDir + projectID + '/'
        self.rewriteFlag = rewriteFlag
                
        # Make local directory files if necessary
        if not os.path.exists(self.localMasterDirectory):
            os.makedirs(self.localMasterDirectory)

        self.startTime = datetime.datetime.now()

        # Download, parse logfile, and verify projectIDs match
        self.logfile = self.localMasterDirectory + 'Logfile.txt'
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + 'Logfile.txt', self.localMasterDirectory])
        self.lp = LP(self.logfile)
        if self.lp.projectID != self.projectID:
            print('ProjectID from logfile: ' + self.lp.projectID + 'does not match projectID folder: ' + self.projectID, file = sys.stderr)
            raise Exception
        
        #Print out some useful info to the user
        print('Beginning analysis of: ' + self.projectID + ' taken from tank: ' + self.lp.tankID, file = sys.stderr)
        print(str(len(self.lp.frames)) + ' total frames for this project from ' + str(self.lp.frames[0].time) + ' to ' + str(self.lp.frames[-1].time), file = sys.stderr)
        print(str(len(self.lp.movies)) + ' total videos for this project.', file = sys.stderr)
        if self.rewriteFlag:
            self._print('Requested data will be reanalyzed from start to finish', file = sys.stderr)
        
        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')

        # Create Depth object (low overhead even if video is just processed)
        self.depthObj = DP(self.localMasterDirectory, self.cloudMasterDirectory, self.logfile)

    def __del__(self):
        # Remove local files once object is destroyed
        #shutil.rmtree(self.localMasterDirectory)
        print('Deleting')
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False
        
    def identifyTray(self):
        if self.rewriteFlag:
            self.depthObj.createTray()
        else:
            self.depthObj.loadTray()
        
    def processDepth(self):
        if self.rewriteFlag:
            self.depthObj.createTray()
            self.depthObj.createSmoothedArray()
            self.depthObj.createBowerLocations()
        else:
            self.depthObj.loadTray()
            self.depthObj.loadSmoothedArray()
            self.depthObj.loadBowerLocations()

        self.depthObj.createDataSummary()
        
    def processVideos(self, index = None):
        if index is None:
            vos = self.lp.movies
        else:
            vos = []
            for i in index:
                vos.append(self.lp.movies[i-1])

        for vo in vos:
            with VP(vo.mp4_file, self.localMasterDirectory, self.cloudMasterDirectory) as vp_obj:
                #vp_obj.createHMM()
                #vp_obj.createClusterHMM()
                vp_obj.createClusterClips()
                vp_obj.cleanup()
                #shutil.rmtree(self.locAnalysisDir + baseName)

    def labelVideos(self, index = None):
        if index is None:
            vos = self.lp.movies
        else:
            vos = []
            for i in index:
                vos.append(self.lp.movies[i-1])

        for vo in vos:
            with VP(vo.mp4_file, self.localMasterDirectory, self.cloudMasterDirectory) as vp_obj:
                vp_obj.labelClusters()
                vp_obj.cleanup()

    def cleanup(self):
        shutil.rmtree(self.localMasterDirectory)

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


