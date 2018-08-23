import os, cv2, sys, datetime, shutil, pims, seaborn
from random import randint
import scipy.signal
import numpy as np
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
            
        # Download, parse logfile, and verify projectIDs match
        subprocess.call(['rclone', 'copy', remote + ':' + self.remMasterDir + 'Logfile.txt', self.locMasterDir])
        self.lp = LP.LogParser(self.locMasterDir + 'Logfile.txt')
        if self.lp.projectID != self.projectID:
            print('ProjectID from logfile: ' + self.lp.projectID + 'does not match projectID folder: ' + self.projectID, file = sys.stderr)
            sys.exit()

        if videos == [0]:
            self.videos = []
        elif videos == [-1]:
            self.videos = [self.lp.movies]
        else:
            self.videos = [self.lp.movies[int(x) - 1] for x in videos]
            
        #Print out some useful info to the user
        print('Beginning analysis of: ' + self.projectID + ' taken from tank: ' + self.lp.tankID, file = sys.stderr)
        print(str(len(self.lp.frames)) + ' total frames for this project from ' + str(self.lp.frames[0].time) + ' to ' + str(self.lp.frames[-1].time), file = sys.stderr)
        print(str(len(self.lp.movies)) + ' total videos for this project, will analyze ' + str(len(self.videos)) + ' of them.', file = sys.stderr)
        if self.rewriteFlag:
            print('All data will be reanalyzed from start to finish', file = sys.stderr)

        # Name of analysis files that will be created
        #self.rawDepthDataFile = self.locAnalysisDir + 'rawDepthData.npy'
        #self.interpDepthDataFile = self.locAnalysisDir + 'interpDepthData.npy'
        self.smoothDepthFile = 'smoothedDepthData.npy'
        self.trayFile = 'TrayInfo.txt'
        self.bowerFile = 'BowerInfo.txt'
        self.transMFile = 'TransformationMatrix.npy'

        # Prepare data - these three functions require user input and should be run first
        self._identifyTray()
        self._identifyBower()
        self._registerImages()        
        
    def processDepth(self):
        self._createSmoothDepthArray()
        self._dailySummary(self.smoothDepthData, 'Smooth')
        #plt.clf()
        #plt.plot(c1)
        #plt.plot(c2)
        #plt.plot(c3)
        #plt.savefig(self.smoothedHourly)
 
    def processVideo(self, index = None):
        if index is None:
            vos = self.lp.movies
        else:
            vos = [self.lp.movies[index]]
        for vo in vos:
            if not os.path.isfile(vo.mp4_file):
                subprocess.call(['ffmpeg', '-framerate', str(vo.framerate),'-i', vo.h264_file, '-c:v', 'copy', vo.mp4_file])
            if os.path.isfile(vo.h264_file):
                if os.stat(vo.mp4_file).st_size >= os.stat(vo.h264_file).st_size:
                    try:
                        vid = pims.Video(vo.mp4_file)
                        vid.close()
                        os.remove(vo.h264_file)
                    except Exception as e:
                        print(e)
                        continue
            self.vp_obj = VideoProcessor(vo, self.analysisDirectory)
            #vp_obj.calculateHMM()

    def _identifyTray(self):

        # If tray attribute already exists, exit
        try:
            self.tray_r
            return
        except AttributeError:
            pass

        # First deterimine if we want to recreate the tray region or try to use what already exists
        if not self.rewriteFlag:
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.trayFile, self.locAnalysisDir])

            if os.path.isfile(self.locAnalysisDir + self.trayFile):
                print('Loading tray information from file on dropbox', file = sys.stderr)
                with open(self.locAnalysisDir + self.trayFile) as f:
                    line = next(f)
                    tray = line.rstrip().split(',')
                    self.tray_r = [int(x) for x in tray]
                return

        # Unable to load it from existing file, either because it doesn't exist or the 
        print('Identify the parts of the frame that include tray to analyze', file = sys.stderr)
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[-1].pic_file, self.locMasterDir + self.lp.frames[-1].frameDir])
        if os.path.isfile(self.locMasterDir + self.lp.frames[-1].pic_file):
            final_image = cv2.imread(self.locMasterDir + self.lp.frames[-1].pic_file,0)
        else:
            print(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[0].npy_file, self.locMasterDir])
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[0].npy_file, self.locMasterDir + self.lp.frames[0].frameDir])
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[-1].npy_file, self.locMasterDir + self.lp.frames[-1].frameDir])
            if not os.path.isfile(self.locMasterDir + self.lp.frames[0].npy_file) or not os.path.isfile(self.locMasterDir + self.lp.frames[-1].npy_file):
                print('Cant find files needed to find the tray! Quitting', file = sys.stderr)
                sys.exit()
            cmap = plt.get_cmap('jet')
            final_image = cmap(plt.Normalize(-10,10)(npy.load(self.locMasterDir + self.lp.frames[-1].npy_file) -  npy.load(self.locMasterDir + self.lp.frames[0].npy_file)))

        cv2.imshow('Identify the parts of the frame that include tray to analyze', final_image)
        tray_r = cv2.selectROI('Identify the parts of the frame that include tray to analyze', final_image, fromCenter = False)
        tray_r = tuple([int(x) for x in tray_r]) # sometimes a float is returned
        self.tray_r = [tray_r[1],tray_r[0],tray_r[1] + tray_r[3], tray_r[0] + tray_r[2]] # (x0,y0,xf,yf)
        # if bounding box is close to edge just set it as the edge
        if self.tray_r[0] < 20: 
            self.tray_r[0] = 0
        if self.tray_r[1] < 20: 
            self.tray_r[1] = 0
        if final_image.shape[0] - self.tray_r[2]  < 20: 
            self.tray_r[2] = final_image.shape[0]
        if final_image.shape[1] - self.tray_r[3]  < 20:  
            self.tray_r[3] = final_image.shape[1]
                
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        with open(self.locAnalysisDir + self.trayFile, 'w') as f:
            print(','.join([str(x) for x in self.tray_r]), file = f)

        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.trayFile, self.remote + ':' + self.remAnalysisDir])


    def _identifyBower(self):

        # If tray attribute already exists, exist
        try:
            self.bowerMask
            return
        except AttributeError:
            pass

       # First deterimine if we want to recreate the tray region or try to use what already exists
        if not self.rewriteFlag:
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.bowerFile, self.locAnalysisDir])

            if os.path.isfile(self.locAnalysisDir + self.bowerFile):
                print('Loading bower information from file on dropbox', file = sys.stderr)
                with open(self.locAnalysisDir + self.bowerFile) as f:
                    line = next(f)
                    tray = line.rstrip().split(',')
                    self.bowerMask = [int(x) for x in tray]
                    return
            
        # Unable to load it from existing file, either because it doesn't exist or the 
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[0].npy_file, self.locMasterDir + self.lp.frames[0].frameDir])
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[-1].npy_file, self.locMasterDir + self.lp.frames[-1].frameDir])
        if not os.path.isfile(self.locMasterDir + self.lp.frames[-1].npy_file) or not os.path.isfile(self.locMasterDir + self.lp.frames[0].npy_file):
            print('Cant find the files to find the bower! Quitting', file = sys.stderr)
            sys.exit()
            
        cmap = plt.get_cmap('jet')
        final_image = cmap(plt.Normalize(-10,10)(np.load(self.locMasterDir + self.lp.frames[0].npy_file) -  np.load(self.locMasterDir + self.lp.frames[-1].npy_file)))
                        
        print('Identifying the bower', file = sys.stderr)
        cv2.imshow('Select region that contains the bower (be generous)', final_image)
        bower_r = cv2.selectROI('Select region that contains the bower (be generous)', final_image, fromCenter = False)
        bower_r = [int(x) for x in bower_r] # sometimes a float is returned
        bower_r = [bower_r[1],bower_r[0],bower_r[1] + bower_r[3], bower_r[0] + bower_r[2]] #(x0,y0,xf,yf)

        if bower_r[0] < 20: 
            bower_r[0] = 0
        if bower_r[1] < 20: 
            bower_r[1] = 0
        if final_image.shape[0] - bower_r[2]  < 20: 
            bower_r[2] = final_image.shape[0]
        if final_image.shape[1] - bower_r[3]  < 20:  
            bower_r[3] = final_image.shape[1]

        self.bowerMask = bower_r
                
        cv2.destroyAllWindows()
        cv2.waitKey(1)
            
        with open(self.locAnalysisDir + self.bowerFile, 'w') as f:
            print(','.join([str(x) for x in self.bowerMask]), file = f)
            
        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.bowerFile, self.remote + ':' + self.remAnalysisDir])

    def _registerImages(self):

        # If tansformation matrix attribute already exists, exist
        try:
            self.transM
            return
        except AttributeError:
            pass

       # First deterimine if we want to recreate the tray region or try to use what already exists
        if not self.rewriteFlag:
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.transMFile, self.locAnalysisDir])

            if os.path.isfile(self.locAnalysisDir + self.transMFile):
                print('Loading transformation matrix information from file on dropbox', file = sys.stderr)
                self.transM = np.load(self.locAnalysisDir + self.transMFile)
                return
                
        # Unable to load it from existing file, either because it doesn't exist or the 
        print('Trying to register RGB and Depth data ', file = sys.stderr)
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.frames[0].pic_file, self.locMasterDir + self.lp.frames[0].frameDir])
        subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + self.lp.movies[0].pic_file, self.locMasterDir + self.lp.movies[0].frameDir])
        if not os.path.isfile(self.locMasterDir + self.lp.frames[0].pic_file) or not os.path.isfile(self.locMasterDir + self.lp.movies[0].pic_file):
            print('Cant find RGB pictures of both ', file = sys.stderr)
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

        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.transMFile, self.remote + ':' + self.remAnalysisDir])

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
            subprocess.call(['rclone', 'copy', self.remote + ':' + self.remAnalysisDir + self.smoothDepthFile, self.locAnalysisDir])

            if os.path.isfile(self.locAnalysisDir + self.smoothDepthFile):
                print('Loading depth information from file on dropbox', file = sys.stderr)
                self.smoothDepthData = np.load(self.locAnalysisDir + self.smoothDepthFile)
                return
        
        #Load or calculate raw data
        print('Creating large npy array to hold raw depth data', file = sys.stderr)
        rawDepthData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
        frameDirectories = set()
        for i, frame in enumerate(self.lp.frames):
            if frame.frameDir not in frameDirectories:
                print('Downloading ' + frame.frameDir + ' from remote host', file = sys.stderr)
                subprocess.call(['rclone', 'copy', self.remote + ':' + self.remMasterDir + frame.frameDir, self.locMasterDir + frame.frameDir])
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

        self.smoothDepthData = scipy.signal.savgol_filter(interpDepthData, tunits, order, axis = 0, mode = 'mirror')
        np.save(self.locAnalysisDir + self.smoothDepthFile, self.smoothDepthData)
        subprocess.call(['rclone', 'copy', self.locAnalysisDir + self.smoothDepthFile, self.remote + ':' + self.remAnalysisDir])

        
    def _createInitialVideo(self):
        self._identifyBowers()
        
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,4)
        ax4 = fig.add_subplot(2,3,5)
        ax5 = fig.add_subplot(2,3,3)
        ax6 = fig.add_subplot(2,3,6)

        v_min = np.nanpercentile(self.rawDepthData, 0.1)
        v_max = np.nanpercentile(self.rawDepthData, 99.9)

        data_o = self.rawDepthData
        data_s_0 = self.smoothDepthData[0]
        
        img_0 = img.imread(self.lp.frames[0].pic_file)
        ax3.imshow(img_0)

        hm_0 = data_o[0]
        ax1.imshow(hm_0, vmin = v_min, vmax = v_max)

        hm_1 = data_s_0
        ax2.imshow(hm_1, vmin = v_min, vmax = v_max)

        hm_2 = data_s_0 - data_s_0
        ax4.imshow(hm_2, vmin = -10, vmax = 10)

        ax5.imshow(self.pit_mask)
        ax6.imshow(self.castle_mask)
        
        writer = animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Patrick McGrath'), bitrate=1800)
        anim = animation.FuncAnimation(fig, self._animate, frames = self.rawDepthData.shape[0], fargs = (fig, ax1, ax2, ax3, ax4, v_min, v_max), repeat = False)
        anim.save(self.analysisVideo, writer = writer)        

    def _animate(self, i, fig, ax1, ax2, ax3, ax4, v_min, v_max):
        print(str(i) + ' of ' + str(self.rawDepthData.shape[0]))
        data_o = self.rawDepthData
        data_s = self.smoothDepthData
        data_s_0 = self.smoothDepthData[0]
        fig.clf()
        

        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,4)
        ax4 = fig.add_subplot(2,3,5)
        ax5 = fig.add_subplot(2,3,3)
        ax6 = fig.add_subplot(2,3,6)
        
        img_0 = img.imread(self.lp.frames[i].pic_file)
        ax3.imshow(img_0)

        hm_0 = data_o[i]
        ax1.imshow(hm_0, vmin = v_min, vmax = v_max)

        hm_1 = data_s[i]
        ax2.imshow(hm_1, vmin = v_min, vmax = v_max)

        hm_2 = data_s[i] - data_s_0
        ax4.imshow(hm_2, vmin = -10, vmax = 10)

        ax5.imshow(self.pit_mask)
        ax6.imshow(self.castle_mask)

    def _dailySummary(self, data, baseFileName, hour_threshold = .5, day_threshold = .5):

        # Histogram info
        total_bins = np.array(list(range(-10, 11, 1)))*.5
        daily_bins = np.array(list(range(-10, 11, 1)))*.2
        hourly_bins = np.array(list(range(-10, 11, 1)))*.1
        
        total_hist = []
        daily_hist = []
        hourly_hist = []
        
        # Create summary figure
        fig = plt.figure(figsize = (11,8.5)) 
        fig.suptitle(self.projectID)
        
        grid = plt.GridSpec(7, self.lp.numDays*4, wspace=0.02, hspace=0.02)

        rect = patches.Rectangle((self.bowerMask[1],self.bowerMask[0]),self.bowerMask[3] - self.bowerMask[1],self.bowerMask[2] - self.bowerMask[0],linewidth=1,edgecolor='r',facecolor='none')
        rect2 = patches.Rectangle((self.bowerMask[1],self.bowerMask[0]),self.bowerMask[3] - self.bowerMask[1],self.bowerMask[2] - self.bowerMask[0],linewidth=1,edgecolor='r',facecolor='none')
        
        pic_ax = fig.add_subplot(grid[0:2, 0:self.lp.numDays*2])
        ax = pic_ax.imshow(data[-1], cmap = 'jet')
        pic_ax.add_patch(rect)
        pic_ax.set_title('Final depth (cm)')
        pic_ax.set_xticklabels([])
        pic_ax.set_yticklabels([])

        plt.colorbar(ax, ax = pic_ax)
        pic_ax2 = fig.add_subplot(grid[0:2, self.lp.numDays*2:])
        ax2 = pic_ax2.imshow(-1*(data[-1] - data[0]), cmap = 'jet')
        pic_ax2.add_patch(rect2)
        pic_ax2.set_title('Total depth change (cm)')
        pic_ax2.set_xticklabels([])
        pic_ax2.set_yticklabels([])
        plt.colorbar(ax2, ax = pic_ax2)

        current_t = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        bigPixels = []
        volumeChange = []
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
                
            
            frames = [x for x in self.lp.frames if x.time > current_t and x.time < current_t + datetime.timedelta(hours = 24) ]
            if len(frames) == 0:
                data_d = np.zeros(shape = data[0].shape)
            else:
                data_i = data[self.lp.frames.index(frames[0])]
                data_f = data[self.lp.frames.index(frames[-1])]
                data_d = (data_f - data_i)[self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]]*-1
                
            a,b = np.histogram(data_d[~np.isnan(data_d)], bins = daily_bins)
            daily_hist.append(a)
            data_d = np.absolute(data_d)
            data_d[data_d < day_threshold] = 0
            data_d[np.isnan(data_d)] = 0

            bigPixels.append(np.count_nonzero(data_d))
            volumeChange.append(np.nansum(data_d))
            data_d[data_d != 0] = 1

            current_ax[i].imshow((data_f - data_i)[self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]], vmin = -2*day_threshold, vmax = 2*day_threshold)
            current_ax2[i].imshow((data_f - data[0])[self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]], vmin = -2*day_threshold, vmax = 2*day_threshold)
            current_ax3[i].imshow(data_d)

            #Calculate largest changing values
            tdata = (data_f - data_i)[self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]]*-1
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

            current_t += datetime.timedelta(hours = 24)

        current_ax1 = fig.add_subplot(grid[6,0:self.lp.numDays*2])
        current_ax2 = fig.add_subplot(grid[6, self.lp.numDays*2:])

        current_ax1.plot(bigPixels)
        current_ax2.plot(volumeChange)
        
        plt.tight_layout()
        #plt.show()
            
        plt.savefig(self.locOutputDir + 'DailySummary.pdf')
        plt.clf()

        subprocess.call(['rclone', 'copy', self.locOutputDir + 'DailySummary.pdf', self.remote + ':' + self.remOutputDir])

        t0 = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

        if self.lp.numDays == 1:
            fig, ax = plt.subplots(nrows = self.lp.numDays+1, ncols = 24+2, sharex = True, sharey=True, figsize = (11,8.5))
        else:
            fig, ax = plt.subplots(nrows = self.lp.numDays, ncols = 24+2, sharex = True, sharey=True, figsize = (11,8.5))

        wd = self.bowerMask[3] - self.bowerMask[1]
        ht = self.bowerMask[2] - self.bowerMask[0]

        changes = []
        for i in range(0, self.lp.numDays):
            out_data = np.zeros(shape = (24,ht,wd))
            changes.append([])
            for j in range(0,24):

                t1 = t0 + datetime.timedelta(hours = 1)
                frames = [x for x in self.lp.frames if x.time > t0 and x.time < t1]
                
                if len(frames) != 0:
                    data_i = data[self.lp.frames.index(frames[0])][self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]]
                    data_f = data[self.lp.frames.index(frames[-1])][self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]]
                    #data_i[self.castle_mask == False] = np.nan
                    #data_f[self.castle_mask == False] = np.nan
                    out_data[j] = -1*(data_f - data_i)*-1
                    try:
                        a,b = np.histogram(out_data[j][~np.isnan(out_data[j])], bins = hourly_bins)
                    except ValueError:
                        print(hourly_bins)
                        print(j)
                        print(len(out_data[j][~np.isnan(out_data[j])]))
                        print(np.min(out_data[j][~np.isnan(out_data[j])]))
                        print(np.max(out_data[j][~np.isnan(out_data[j])]))

                    hourly_hist.append(a)
                    ax[i,j].imshow(out_data[j], vmin = -2*hour_threshold, vmax = 2*hour_threshold)
                    tdata = out_data[j].copy()
                    gd = tdata[~np.isnan(tdata)]
                    big_pix = len(gd[(gd < -1*hour_threshold) | (gd > hour_threshold)])/len(gd)
                    sum_pix = np.abs(gd[(gd < -1*hour_threshold) | (gd > hour_threshold)]).sum()
                    changes[i].append(sum_pix)
                    #build_data[i,j] = castle_total[self.lp.frames.index(frames[-1])] - castle_total[self.lp.frames.index(frames[0])]
                else:
                    #a = (castle_ind[0].min(),castle_ind[0].max(), castle_ind[1].min(),castle_ind[1].max())
                    zerodata = np.zeros(shape = (1600,1600))
                    ax[i,j].imshow(np.zeros(shape = (1600,1600))[self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]], vmin = -2*hour_threshold, vmax = 2*hour_threshold)
                    changes[i].append(0)
                ax[i,j].set_adjustable('box-forced')
                ax[i,j].set_xticklabels([])
                ax[i,j].set_yticklabels([])


                t0 = t1

            frames = [x for x in self.lp.frames if x.time > t1 - datetime.timedelta(hours = 24) and x.time < t1]
            if len(frames) == 0:
                out_data = np.zeros(shape = data[0].shape)
            else:
                data_i = data[self.lp.frames.index(frames[0])][self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]]
                data_f = data[self.lp.frames.index(frames[-1])][self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]]
                out_data = (data_f - data_i)
            gd = out_data[~np.isnan(out_data)]
            gd = out_data[~np.isnan(out_data)]
            big_pix = len(gd[(gd < -5) | (gd > 5)])/len(gd)

            ax[i,j+1].imshow(out_data, vmin = -2*day_threshold, vmax = 2*day_threshold)
            ax[i,j+1].set_adjustable('box-forced')

            data_i = data[0][self.bowerMask[0]:self.bowerMask[2],self.bowerMask[1]:self.bowerMask[3]]
            out_data = (data_f - data_i)
            ax[i,j+2].imshow(out_data, vmin = -4*day_threshold, vmax = 4*day_threshold)
            ax[i,j+2].set_adjustable('box-forced')

                #        plt.imshow(build_data)
                #        plt.colorbar()
        plt.subplots_adjust(wspace = 0, hspace = 0)

        plt.savefig(self.locOutputDir + 'HourlySummary.pdf')
        subprocess.call(['rclone', 'copy', self.locOutputDir + 'HourlySummary.pdf', self.remote + ':' + self.remOutputDir])

        plt.clf()

        seaborn.heatmap(changes)
        plt.savefig(self.locOutputDir + 'HourlyHeatmap.pdf')
        subprocess.call(['rclone', 'copy', self.locOutputDir + 'HourlyHeatmap.pdf', self.remote + ':' + self.remOutputDir])
        plt.clf()
        a,b = np.histogram(out_data[~np.isnan(out_data)], bins = total_bins)
        total_hist = a
        with open(self.locOutputDir + 'Histograms.txt', 'w') as f:
            print('Total', file = f)
            print('\t'.join([str(x) for x in total_bins]), file = f)
            print('\t'.join([str(x) for x in total_hist]), file = f)
            print('Daily', file = f)
            print('\t'.join([str(x) for x in daily_bins]), file = f)
            for i in range(0,len(daily_hist)):
                print('\t'.join([str(x) for x in daily_hist[i]]), file = f)
            print('Hourly', file = f)
            print('\t'.join([str(x) for x in hourly_bins]), file = f)
            for i in range(0,len(hourly_hist)):
                print('\t'.join([str(x) for x in hourly_hist[i]]), file = f)

