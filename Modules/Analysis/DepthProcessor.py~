import datetime, subprocess, cv2, getpass, socket, os, sys
import scipy.signal

import Modules.LogParser as LP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage import morphology

np.warnings.filterwarnings('ignore')

class DepthProcessor:
    # This class takes in directory information and a logfile containing depth information and performs the following:
    # 1. Identifies tray using manual input
    # 2. Interpolates and smooths depth data
    # 3. Automatically identifies bower location
    # 4. Analyze building, shape, and other pertinent info of the bower

    def __init__(self, localMasterDirectory, cloudMasterDirectory, logfile):

        self.localMasterDirectory = localMasterDirectory if localMasterDirectory[-1] == '/' else localMasterDirectory + '/'
        self.cloudMasterDirectory = cloudMasterDirectory if cloudMasterDirectory[-1] == '/' else cloudMasterDirectory + '/'
        self.lp = LP.LogParser(logfile)
        self.localDepthDirectory = self.localMasterDirectory + 'DepthAnalysis/'
        self.cloudDepthDirectory = self.cloudMasterDirectory + 'DepthAnalysis/'

        # Make local temp directory if it doesnt exist
        if not os.path.exists(self.localDepthDirectory):
            os.makedirs(self.localDepthDirectory)
        
        self.trayFile = 'trayInfo.txt'
        self.interpDepthFile = 'interpDepthData.npy'
        self.smoothDepthFile = 'smoothedDepthData.npy'
        self.bowerLocationFile = 'bowerLocations.npy'
        self.histogramFile = 'dataHistograms.xlsx'

        self.anLF = open(self.localDepthDirectory + 'DepthAnalysisLog.txt', 'a')
        self.fnull = open(os.devnull, 'a')
        
    def createTray(self):
        
        # Log info 
        self._print('Manual identification of the tray from depth data')

        # Download first and last depth data frame to use for tray identification
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.lp.frames[0].npy_file, self.localMasterDirectory + self.lp.frames[0].frameDir], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.lp.frames[-1].npy_file, self.localMasterDirectory + self.lp.frames[-1].frameDir], stderr = self.fnull)
        if not os.path.isfile(self.localMasterDirectory + self.lp.frames[0].npy_file) or not os.path.isfile(self.localMasterDirectory + self.lp.frames[-1].npy_file):
            self._print('Cant find files needed to find the tray!')
            raise Exception

        # Create color image of depth change
        cmap = plt.get_cmap('jet')
        final_image = cmap(plt.Normalize(-10,10)(np.load(self.localMasterDirectory + self.lp.frames[-1].npy_file) -  np.load(self.localMasterDirectory + self.lp.frames[0].npy_file)))

        # Query user to identify regions of the tray that are good
        cv2.imshow('Identify the parts of the frame that include tray to analyze', final_image)
        tray_r = cv2.selectROI('Identify the parts of the frame that include tray to analyze', final_image, fromCenter = False)
        tray_r = tuple([int(x) for x in tray_r]) # sometimes a float is returned
        self.tray_r = [tray_r[1],tray_r[0],tray_r[1] + tray_r[3], tray_r[0] + tray_r[2]] # (x0,y0,xf,yf)
        
        # if bounding box is close to the edge just set it as the edge
        if self.tray_r[0] < 50: 
            self.tray_r[0] = 0
        if self.tray_r[1] < 50: 
            self.tray_r[1] = 0
        if final_image.shape[0] - self.tray_r[2]  < 50: 
            self.tray_r[2] = final_image.shape[0]
        if final_image.shape[1] - self.tray_r[3]  < 50:  
            self.tray_r[3] = final_image.shape[1]

        # Destroy windows (running it 3 times helps for some reason)
        for i in range(3):
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        # Save and back up tray file
        with open(self.localDepthDirectory + self.trayFile, 'w') as f:
            print(','.join([str(x) for x in self.tray_r]), file = f)

        self._cloudUpdate()

    def createSmoothedArray(self, totalGoodData = 0.3, minGoodData = 0.5, minUnits = 5, tunits = 71, order = 4, rewrite = False):
        # Get tray info
        self.loadTray()
        
        # Download raw data and create new array to store it
        self._print('Calculating smoothed depth array from raw data')
        rawDepthData = np.empty(shape = (len(self.lp.frames), self.lp.height, self.lp.width))
        frameDirectories = set()
        for i, frame in enumerate(self.lp.frames):
            if frame.frameDir not in frameDirectories:
                print('Downloading ' + frame.frameDir + ' from remote host', file = sys.stderr)
                subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + frame.frameDir, self.localMasterDirectory + frame.frameDir, '--exclude', '*.mp4', '--exclude', '*.h264'])
                frameDirectories.add(frame.frameDir)
                
            try:
                data = np.load(self.localMasterDirectory + frame.npy_file)
            except ValueError:
                self._print('Bad frame: ' + str(i) + ', ' + frame.npy_file)
                rawDepthData[i] = self.rawDepthData[i-1]
            else:
                rawDepthData[i] = data

        # Convert to cm
        rawDepthData = 100/(-0.0037*rawDepthData + 3.33)

        rawDepthData[(rawDepthData < 40) | (rawDepthData > 80)] = np.nan # Values that are too close or too far are set to np.nan

        # Make copy of raw data
        interpDepthData = rawDepthData.copy()

        # Count number of good pixels
        goodDataAll = np.count_nonzero(~np.isnan(interpDepthData), axis = 0) # number of good data points per pixel
        goodDataStart = np.count_nonzero(~np.isnan(interpDepthData[:100]), axis = 0) # number of good data points in the first 5 hours

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

        np.save(self.localDepthDirectory + self.interpDepthFile, interpDepthData)
        print('Smoothing data with savgol filter', file = sys.stderr)
        self.smoothDepthData = scipy.signal.savgol_filter(interpDepthData, tunits, order, axis = 0, mode = 'mirror')
        np.save(self.localDepthDirectory + self.smoothDepthFile, self.smoothDepthData)

        self._cloudUpdate()

    def createBowerLocations(self, totalThreshold = 1.0, dayThreshold = 0.4, minPixels = 500):

        self._print('Identifying bower locations from depth data')

        self.loadTray()
        self.loadSmoothedArray() # Load depth data if necessary

        self.bowerLocations = np.zeros(shape = (self.lp.numDays + 1, self.lp.height, self.lp.width), dtype = "int8")
        print(self.lp.numDays)
        print(self.bowerLocations.shape)
        
        bins = np.array(list(range(-100, 101, 1)))*.2 # for getting histogram info
        self.histograms = pd.DataFrame(index = bins[0:-1]) # for getting histogram info

        #TotalChange
        tFirst = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        tLast = self.lp.frames[-1].time.replace(hour = 23, minute = 59, second = 59, microsecond = 999999)
        tChange = self._returnHeightChange(tFirst, tLast)

        tCastle = tChange.copy()
        tCastle[tCastle < totalThreshold] = 0
        tCastle[np.isnan(tCastle)] = 0
        tCastle[tCastle!=0] = 1
        tCastle = morphology.remove_small_objects(tCastle.astype(bool), minPixels)

        tPit = tChange.copy()
        tPit[tPit > -1*totalThreshold] = 0
        tPit[np.isnan(tPit)] = 0
        tPit[tPit!=0] = 1
        tPit = morphology.remove_small_objects(tPit.astype(bool), minPixels)

        self.bowerLocations[0][tCastle == True] = 1
        self.bowerLocations[0][tPit == True] = -1

        # Histogram information
        a,b = np.histogram(tChange[~np.isnan(tChange)], bins = bins)
        self.histograms['Total'] = pd.Series(a, index = bins[:-1])

        #DailyChange        
        for i in range(self.lp.numDays):
            
            start = tFirst + datetime.timedelta(hours = 24*i)
            end = tFirst + datetime.timedelta(hours = 24*(i+1))

            tChange = self._returnHeightChange(start, end)
            tCastle = tChange.copy()
            tCastle[tCastle < dayThreshold] = 0
            tCastle[np.isnan(tCastle)] = 0
            tCastle[tCastle!=0] = 1
            tCastle = morphology.remove_small_objects(tCastle.astype(bool), minPixels)

            tPit = tChange.copy()
            tPit[tPit > -1*dayThreshold] = 0
            tPit[np.isnan(tPit)] = 0
            tPit[tPit!=0] = 1
            tPit = morphology.remove_small_objects(tPit.astype(bool), minPixels)

            self.bowerLocations[i+1][tCastle == True] = 1
            self.bowerLocations[i+1][tPit == True] = -1

            a,b = np.histogram(tChange[~np.isnan(tChange)], bins = bins)
            self.histograms['Day' + str(i+1)] = pd.Series(a, index = bins[:-1])

        np.save(self.localDepthDirectory + self.bowerLocationFile, self.bowerLocations)

        writer = pd.ExcelWriter(self.localDepthDirectory + self.histogramFile)
        self.histograms.to_excel(writer, 'Histogram')
        writer.save()
        
        self._cloudUpdate()

    def createDataSummary(self, dayThreshold = 0.4):
        self._print('Creating Data Summary')

        self.loadTray()
        self.loadSmoothedArray()
        self.loadBowerLocations()
        
        # Create summary figure of daily values
        fig = plt.figure(figsize = (11,8.5)) 
        fig.suptitle(self.lp.projectID)
        
        grid = plt.GridSpec(10, self.lp.numDays*4, wspace=0.02, hspace=0.02)

        start_day = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

        # Show picture of final depth
        pic_ax = fig.add_subplot(grid[0:2, 0:self.lp.numDays*1])
        ax = pic_ax.imshow(self._returnHeightChange(-1, tray = True), vmin = 50, vmax = 70)
        pic_ax.set_title('Final depth (cm)')
        pic_ax.set_xticklabels([])
        pic_ax.set_yticklabels([])
        plt.colorbar(ax, ax = pic_ax)

        # Show picture of total depth change
        pic_ax2 = fig.add_subplot(grid[0:2, self.lp.numDays*1:self.lp.numDays*2])
        ax2 = pic_ax2.imshow(self._returnHeightChange(start_day, -1, tray = True), vmin = -5, vmax = 5)
        pic_ax2.set_title('Depth change (cm)')
        pic_ax2.set_xticklabels([])
        pic_ax2.set_yticklabels([])
        plt.colorbar(ax2, ax = pic_ax2)

        pic_ax3 = fig.add_subplot(grid[0:2, self.lp.numDays*2:self.lp.numDays*3])
        ax3 = pic_ax3.imshow(self._returnMask(tray = True), vmin = -1, vmax = 1)
        pic_ax3.set_title('Mask')
        pic_ax3.set_xticklabels([])
        pic_ax3.set_yticklabels([])
        plt.colorbar(ax3, ax = pic_ax3)

        pic_ax4 = fig.add_subplot(grid[0:2, self.lp.numDays*3:self.lp.numDays*4])
        tdata = self._returnHeightChange(start_day, start_day + datetime.timedelta(hours = 24*self.lp.numDays), tray = True)
        tdata[np.isnan(tdata)] = 0
        x_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_values = []
        for thresh in x_values:
            tdata[(tdata<thresh) & (tdata > -1*thresh)] = 0
            y_values.append(tdata.sum())
        print(x_values)
        print(y_values)
        pic_ax4.plot(x_values, y_values)  
            
        bigPixels = []
        volumeChange = []

        #DailyChange        
        for i in range(self.lp.numDays):
            tdata[(tdata<dayThreshold*thresh) & (tdata > dayThreshold*-1*thresh)] = 0
            
            if i == 0:
                current_ax = [fig.add_subplot(grid[3, i*4:i*4+3])]
                current_ax2 = [fig.add_subplot(grid[4, i*4:i*4+3], sharex = current_ax[i])]
                current_ax3 = [fig.add_subplot(grid[5, i*4:i*4+3], sharex = current_ax[i])]
                current_ax4 = [fig.add_subplot(grid[6, i*4:i*4+3])]
                current_ax5 = [fig.add_subplot(grid[7, i*4:i*4+3])]
                current_ax6 = [fig.add_subplot(grid[8, i*4:i*4+3])]

            else:
                current_ax.append(fig.add_subplot(grid[3, i*4:i*4+3], sharey = current_ax[0]))
                current_ax2.append(fig.add_subplot(grid[4, i*4:i*4+3], sharex = current_ax[i], sharey = current_ax2[0]))
                current_ax3.append(fig.add_subplot(grid[5, i*4:i*4+3], sharex = current_ax[i], sharey = current_ax3[0]))
                current_ax4.append(fig.add_subplot(grid[6, i*4:i*4+3], sharey = current_ax4[0]))
                current_ax5.append(fig.add_subplot(grid[7, i*4:i*4+3], sharey = current_ax5[0]))
                current_ax6.append(fig.add_subplot(grid[8, i*4:i*4+3], sharey = current_ax6[0]))

            start = start_day + datetime.timedelta(hours = 24*i)
            stop = start_day + datetime.timedelta(hours = 24*(i+1))

            
            

            
            current_ax[i].set_title('Day ' + str(i+1))

            current_ax[i].imshow(self._returnHeightChange(start_day, stop, tray = True), vmin = -4*dayThreshold, vmax = 4*dayThreshold)
            current_ax2[i].imshow(self._returnHeightChange(start, stop, tray = True), vmin = -4*dayThreshold, vmax = 4*dayThreshold)
            current_ax3[i].imshow(self._returnHeightChange(start, stop, masked = True, tray = True), vmin = -4*dayThreshold, vmax = 4*dayThreshold)

            bigPixels.append(np.count_nonzero(self._returnMask(start, tray = True)))
            volumeChange.append(np.nansum(np.absolute(self._returnHeightChange(start, stop, tray = True))))
   
           
            #Calculate largest changing values
            tdata = self._returnHeightChange(start, stop, tray = True, masked = True)
            volumeChange.append(tdata.sum())
            bigPixels.append(np.count_nonzero(tdata))
            x_values = [1, 1.5, 2, 2.5, 3.0]
            y_values = []
            n_values = []
            b_values = []
            for thresh in x_values:
                tdata[(tdata<dayThreshold*thresh) & (tdata > dayThreshold*-1*thresh)] = 0
                
                y_values.append(tdata.sum())
                n_values.append(np.count_nonzero(tdata))
                b_values.append(tdata.sum()/np.count_nonzero(tdata))
                
            current_ax4[i].plot(x_values, y_values)
            current_ax5[i].plot(x_values, n_values)
            current_ax6[i].plot(x_values, b_values)
                            
            
            current_ax[i].set_xticklabels([])
            current_ax2[i].set_xticklabels([])
            current_ax3[i].set_xticklabels([])

            #if i != 0:
            #    current_ax4[i].set_yticklabels([])
            #    current_ax5[i].set_yticklabels([])
            #    current_ax6[i].set_yticklabels([])

            current_ax[i].set_yticklabels([])
            current_ax2[i].set_yticklabels([])
            current_ax3[i].set_yticklabels([])
            current_ax[i].set_adjustable('box-forced')
            current_ax2[i].set_adjustable('box-forced')
            current_ax3[i].set_adjustable('box-forced')

        #current_ax1 = fig.add_subplot(grid[6,0:self.lp.numDays*2])
        #current_ax2 = fig.add_subplot(grid[6, self.lp.numDays*2:])

        #current_ax1.plot(bigPixels)
        #current_ax2.plot(volumeChange)
        
        plt.tight_layout()
        #plt.show()
            
        plt.savefig(self.localDepthDirectory + 'DailySummary.pdf')
        plt.clf()

        self._cloudUpdate()

        fig = plt.figure(figsize = (11,8.5)) 
        fig.suptitle(self.lp.projectID)
        
        grid = plt.GridSpec(self.lp.numDays, 14, wspace=0.02, hspace=0.02)

        changes = []
        start_day = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

        for i in range(0, self.lp.numDays):
            for j in range(12):
                start = start_day + datetime.timedelta(hours = 24*i + j*2)
                stop = start_day + datetime.timedelta(hours = 24*i + (j+1)*2)

                current_ax = fig.add_subplot(grid[i, j])

                current_ax.imshow(self._returnHeightChange(start, stop, tray = True), vmin = -2*dayThreshold, vmax = 2*dayThreshold)
                current_ax.set_adjustable('box-forced')
                current_ax.set_xticklabels([])
                current_ax.set_yticklabels([])
                if i == 0:
                    current_ax.set_title(str(j*2) + '-' + str((j+1)*2))

            current_ax = fig.add_subplot(grid[i, 12])
            current_ax.imshow(self._returnMask(start, tray = True), vmin = -1, vmax = 1)
            current_ax.set_adjustable('box-forced')
            current_ax.set_xticklabels([])
            current_ax.set_yticklabels([])
            if i==0:
                current_ax.set_title('Mask')


            current_ax = fig.add_subplot(grid[i, 13])
            current_ax.imshow(self._returnHeightChange(stop - datetime.timedelta(hours = 24), stop, tray = True), vmin = -2*dayThreshold, vmax = 2*dayThreshold)
            current_ax.set_adjustable('box-forced')
            current_ax.set_xticklabels([])
            current_ax.set_yticklabels([])
            if i==0:
                current_ax.set_title('DailyChange')


            
        plt.savefig(self.localDepthDirectory + 'HourlySummary.pdf')
        self._cloudUpdate()
        plt.clf()
        
    def loadTray(self):
        # If tray attribute already exists, exit
        try:
            self.tray_r
            return
        except AttributeError:
            pass

        # Try to find tray file on cloud directory
        subprocess.call(['rclone', 'copy', self.cloudDepthDirectory + self.trayFile, self.localDepthDirectory], stderr = self.fnull)

        if os.path.isfile(self.localDepthDirectory + self.trayFile):
            print('Loading tray information from file on cloud directory', file = sys.stderr)
            with open(self.localDepthDirectory + self.trayFile) as f:
                line = next(f)
                tray = line.rstrip().split(',')
                self.tray_r = [int(x) for x in tray]
            return
        else:
            self.createTray() # Trayfile doesn't exist, need to create it

    def loadSmoothedArray(self):
        try:
            self.smoothDepthData
            return
        except AttributeError:
            pass

        if not os.path.isfile(self.localDepthDirectory + self.smoothDepthFile):
            print('Downloading depth file from Dropbox', file = sys.stderr)
            subprocess.call(['rclone', 'copy', self.cloudDepthDirectory + self.smoothDepthFile, self.localDepthDirectory], stderr = self.fnull)
        if os.path.isfile(self.localDepthDirectory + self.smoothDepthFile):
            self.smoothDepthData = np.load(self.localDepthDirectory + self.smoothDepthFile)
        else:
            self.createSmoothedArray() # Needs to be created

    def loadBowerLocations(self):
        try:
            self.bowerLocations
            return
        except AttributeError:
            pass

        if not os.path.isfile(self.localDepthDirectory + self.bowerLocationFile):
            print('Downloading depth file from Dropbox', file = sys.stderr)
            subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.bowerLocationFile, self.localDepthDirectory], stderr = self.fnull)
            
        if os.path.isfile(self.localDepthDirectory + self.bowerLocationFile):
            self.bowerLocations = np.load(self.localDepthDirectory + self.bowerLocationFile)
        else:
            self.createBowerLocations()
            
    def _returnHeightChange(self, t0=None, t1 = None, masked = False, tray = False):
        self.loadSmoothedArray() #Make sure array is loaded

        if masked:
            self.loadBowerLocations()

        if tray:
            self.loadTray()
        
        # Find closest frames to desired times
        try:
            if t0 == -1:
                first_index = -1
            else:
                first_index = max([False if x.time<=t0 else True for x in self.lp.frames].index(True) - 1, 0) #This ensures that we get overnight changes when kinect wasn't running
        except ValueError:
            if t0 > self.lp.frames[-1].time:
                first_index = -1
            else:
                first_index = 0

        if t1 is not None:
            try:
                if t1 == -1:
                    last_index = -1
                else:
                    last_index = max([False if x.time<=t1 else True for x in self.lp.frames].index(True) - 1, 0)
            except ValueError:
                last_index = len(self.lp.frames) - 1
            
            change = self.smoothDepthData[first_index] - self.smoothDepthData[last_index]

        else:
            change = self.smoothDepthData[first_index]  
        
        if masked:
            if (t1 - t0).total_seconds() > 86400: # if difference between times is greater than 24 hours use total mask 
                change[self._returnMask() == 0] = 0
            else:
                change[self._returnMask(t0) == 0] = 0

        if tray:
            change = change[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]]
            
        return change

    def _returnMask(self, t1 = None, tray = False):
        self.loadBowerLocations()

        if tray:
            self.loadTray()


        if t1 is None:
            out = self.bowerLocations[0]
        else:
            startTime = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
            index = (t1 - startTime).days + 1
            try:
                out = self.bowerLocations[index]
            except IndexError:
                print(t1)
                print(self.lp.frames[0].time)
                print(index)
                print(t1 - self.lp.frames[0].time)
                raise IndexError

        if tray:
            return out[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]]
        else:
            return out
        
    def _cloudUpdate(self):
        print('Syncing with cloud', file = sys.stderr)
        subprocess.call(['rclone', 'copy', self.localDepthDirectory, self.cloudDepthDirectory], stderr = self.fnull)
        #subprocess.call(['rclone', 'copy', self.localOutputDirectory, self.cloudOutputDirectory], stderr = self.fnull)

        
    def _print(self, outtext):
        print(str(getpass.getuser()) + ' analyzed at ' + str(datetime.datetime.now()) + ' on ' + socket.gethostname() + ': ' + outtext, file = self.anLF)
        print(outtext, file = sys.stderr)
        self.anLF.close() # Close and reopen file to flush it
        self.anLF = open(self.localDepthDirectory + 'DepthAnalysisLog.txt', 'a')

