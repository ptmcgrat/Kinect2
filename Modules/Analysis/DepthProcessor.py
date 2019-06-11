import datetime, subprocess, cv2, getpass, socket, os, sys
import scipy.signal

import Modules.LogParser as LP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage import morphology

from collections import OrderedDict

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
        self.dataSummary = 'summarizedData.xlsx'
        self.dailySummary = 'DailySummary.pdf'
        self.hourlySummary = 'HourlySummary.pdf'
        self.totalHeightChange = 'totalHeightChange.npy'
        self.totalBowerLocation = 'totalBowerLocations.npy'

        self.convertPixel = 0.1030168618 # cm / pixel

        self.anLF = open(self.localDepthDirectory + 'DepthAnalysisLog.txt', 'a')
        self.fnull = open(os.devnull, 'a')
        
    def createTray(self):
        
        # Log info 
        self._print('Created ' + self.trayFile)

        # Download first and last depth data frame to use for tray identification
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.lp.frames[0].npy_file, self.localMasterDirectory + self.lp.frames[0].frameDir], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.lp.frames[-1].npy_file, self.localMasterDirectory + self.lp.frames[-1].frameDir], stderr = self.fnull)
        if not os.path.isfile(self.localMasterDirectory + self.lp.frames[0].npy_file) or not os.path.isfile(self.localMasterDirectory + self.lp.frames[-1].npy_file):
            self._print('Error: Cant find files needed to find the tray!')
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
        self._print('Created ' + self.interpDepthFile)
        self._print('Created ' + self.smoothDepthFile)
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

    def createDataSummary(self, hourlyDelta = 2):

        self._print('Creating ' + self.dataSummary)
        self._print('Creating ' + self.dailySummary)
        self._print('Creating ' + self.hourlySummary)
        self._print('Creating ' + self.totalHeightChange)
        self._print('Creating ' + self.totalBowerLocation)

        np.save(self.localDepthDirectory + self.totalHeightChange, self._returnHeightChange(self.lp.frames[0].time, self.lp.frames[-1].time, cropped = True))
        np.save(self.localDepthDirectory + self.totalBowerLocation, self._returnBowerLocations(self.lp.frames[0].time, self.lp.frames[-1].time, cropped = True))

        # Create summary figure of daily values
        figDaily = plt.figure(figsize = (11,8.5)) 
        figDaily.suptitle(self.lp.projectID + ' DailySummary')
        gridDaily = plt.GridSpec(10, self.lp.numDays*4, wspace=0.02, hspace=0.02)

        # Create summary figure of hourly values
        figHourly = plt.figure(figsize = (11,8.5)) 
        figHourly.suptitle(self.lp.projectID + ' HourlySummary')
        gridHourly = plt.GridSpec(self.lp.numDays, int(24/hourlyDelta) + 2, wspace=0.02, hspace=0.02)


        start_day = self.lp.frames[0].time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        totalChangeData = self._summarizeBuilding(self.lp.frames[0].time, self.lp.frames[-1].time, extremePixels = 2000)

        # Show picture of final depth
        topAx1 = figDaily.add_subplot(gridDaily[0:2, 0:self.lp.numDays*1-1])
        topAx1_ax = topAx1.imshow(self._returnHeight(self.lp.frames[-1].time, cropped = True), vmin = 50, vmax = 70)
        topAx1.set_title('Final depth (cm)')
        topAx1.set_xticklabels([])
        topAx1.set_yticklabels([])
        plt.colorbar(topAx1_ax, ax = topAx1)

        # Show picture of total depth change
        topAx2 = figDaily.add_subplot(gridDaily[0:2, self.lp.numDays*1:self.lp.numDays*2-1])
        topAx2_ax = topAx2.imshow(self._returnHeightChange(self.lp.frames[0].time, self.lp.frames[-1].time, cropped = True), vmin = -5, vmax = 5)
        topAx2.set_title('Total depth change (cm)')
        topAx2.set_xticklabels([])
        topAx2.set_yticklabels([])
        plt.colorbar(topAx2_ax, ax = topAx2)

        # Show picture of pit and castle mask
        topAx3 = figDaily.add_subplot(gridDaily[0:2, self.lp.numDays*2:self.lp.numDays*3-1])
        topAx3_ax = topAx3.imshow(self._returnHeightChange(self.lp.frames[0].time, self.lp.frames[-1].time, cropped = True, masked = True), vmin = -5, vmax = 5)
        topAx3.set_title('Mask')
        topAx3.set_xticklabels([])
        topAx3.set_yticklabels([])
        #plt.colorbar(topAx3_ax, ax = topAx3)

        # Bower index based upon higher thresholds
        topAx4 = figDaily.add_subplot(gridDaily[0:2, self.lp.numDays*3:self.lp.numDays*4])
        x_values = [1.0, 3.0, 5.0]
        y_values = []
        for thresh in x_values:
            tdata = self._summarizeBuilding(self.lp.frames[0].time, self.lp.frames[-1].time, totalThreshold = thresh)
            y_values.append(tdata['bowerIndex'])
            totalChangeData['bowerIndex_' + str(thresh)] = tdata['bowerIndex']
        topAx4.plot(x_values, y_values)
        topAx4.set_title('BowerIndex vs. Threshold (cm)')
        figDaily.tight_layout()    

        # Create figures and get data for daily Changes
        dailyChangeData = []
        for i in range(self.lp.numDays):
            if i == 0:
                current_ax = [figDaily.add_subplot(gridDaily[3, i*4:i*4+3])]
                current_ax2 = [figDaily.add_subplot(gridDaily[4, i*4:i*4+3], sharex = current_ax[i])]
                current_ax3 = [figDaily.add_subplot(gridDaily[5, i*4:i*4+3], sharex = current_ax[i])]
                
            else:
                current_ax.append(figDaily.add_subplot(gridDaily[3, i*4:i*4+3], sharey = current_ax[0]))
                current_ax2.append(figDaily.add_subplot(gridDaily[4, i*4:i*4+3], sharex = current_ax[i], sharey = current_ax2[0]))
                current_ax3.append(figDaily.add_subplot(gridDaily[5, i*4:i*4+3], sharex = current_ax[i], sharey = current_ax3[0]))
                
            start = start_day + datetime.timedelta(hours = 24*i)
            stop = start_day + datetime.timedelta(hours = 24*(i+1))
            
            dailyChangeData.append(self._summarizeBuilding(start,stop, extremePixels = 2000))
            dailyChangeData[i]['Day'] = i+1
            dailyChangeData[i]['Midpoint'] = i+1 + .5
            dailyChangeData[i]['StartTime'] = str(start)
            for thresh in [0.4, 0.8, 1.2]:
               tempData = self._summarizeBuilding(start,stop, totalThreshold = thresh)
               dailyChangeData[i]['bowerIndex_' + str(thresh)] = tempData['bowerIndex']


            current_ax[i].set_title('Day ' + str(i+1))

            current_ax[i].imshow(self._returnHeightChange(start_day, stop, cropped = True), vmin = -2, vmax = 2)
            current_ax2[i].imshow(self._returnHeightChange(start, stop, cropped = True), vmin = -2, vmax = 2)
            current_ax3[i].imshow(self._returnHeightChange(start, stop, masked = True, cropped = True), vmin = -2, vmax = 2)
           
            current_ax[i].set_xticklabels([])
            current_ax2[i].set_xticklabels([])
            current_ax3[i].set_xticklabels([])

            current_ax[i].set_yticklabels([])
            current_ax2[i].set_yticklabels([])
            current_ax3[i].set_yticklabels([])

            current_ax[i].set_adjustable('box-forced')
            current_ax2[i].set_adjustable('box-forced')
            current_ax3[i].set_adjustable('box-forced')

        figDaily.tight_layout()
        hourlyChangeData = []

        for i in range(0, self.lp.numDays):
            for j in range(int(24/hourlyDelta)):
                start = start_day + datetime.timedelta(hours = 24*i + j*hourlyDelta)
                stop = start_day + datetime.timedelta(hours = 24*i + (j+1)*hourlyDelta)

                hourlyChangeData.append(self._summarizeBuilding(start,stop, extremePixels = 2000))
                hourlyChangeData[-1]['Day'] = i+1
                hourlyChangeData[-1]['Midpoint'] = i+1 + ((j + 0.5) * hourlyDelta)/24
                hourlyChangeData[-1]['StartTime'] = str(start)

                for thresh in [0.2, 0.4, 0.8]:
                    tempData = self._summarizeBuilding(start,stop,totalThreshold = thresh)
                    hourlyChangeData[-1]['bowerIndex_' + str(thresh)] = tempData['bowerIndex']

                current_ax = figHourly.add_subplot(gridHourly[i, j])

                current_ax.imshow(self._returnHeightChange(start, stop, cropped = True), vmin = -1, vmax = 1)
                current_ax.set_adjustable('box-forced')
                current_ax.set_xticklabels([])
                current_ax.set_yticklabels([])
                if i == 0:
                    current_ax.set_title(str(j*hourlyDelta) + '-' + str((j+1)*hourlyDelta))

            current_ax = figHourly.add_subplot(gridHourly[i, int(24/hourlyDelta)])
            current_ax.imshow(self._returnBowerLocations(stop - datetime.timedelta(hours = 24), stop, cropped = True), vmin = -1, vmax = 1)
            current_ax.set_adjustable('box-forced')
            current_ax.set_xticklabels([])
            current_ax.set_yticklabels([])
            if i==0:
                current_ax.set_title('DMask')


            current_ax = figHourly.add_subplot(gridHourly[i, int(24/hourlyDelta)+1])
            current_ax.imshow(self._returnHeightChange(stop - datetime.timedelta(hours = 24), stop, cropped = True), vmin = -1, vmax = 1)
            current_ax.set_adjustable('box-forced')
            current_ax.set_xticklabels([])
            current_ax.set_yticklabels([])
            if i==0:
                current_ax.set_title('DChange')

        totalDT = pd.DataFrame([totalChangeData])
        dailyDT = pd.DataFrame(dailyChangeData)
        hourlyDT = pd.DataFrame(hourlyChangeData)

        writer = pd.ExcelWriter(self.localDepthDirectory + self.dataSummary)
        totalDT.to_excel(writer,'Total')
        dailyDT.to_excel(writer,'Daily')
        hourlyDT.to_excel(writer,'Hourly')
        writer.save()

        volAx = figDaily.add_subplot(gridDaily[6:8, 0:self.lp.numDays*4])
        volAx.plot(dailyDT['Midpoint'], dailyDT['totalVolume'])
        volAx.plot(hourlyDT['Midpoint'], hourlyDT['totalVolume'])
        volAx.set_ylabel('Volume Change')

        bIAx = figDaily.add_subplot(gridDaily[8:10, 0:self.lp.numDays*4], sharex = volAx)
        bIAx.scatter(dailyDT['Midpoint'], dailyDT['bowerIndex'])
        bIAx.scatter(hourlyDT['Midpoint'], hourlyDT['bowerIndex'])
        bIAx.set_xlabel('Day')
        bIAx.set_ylabel('Bower Index')



        figDaily.savefig(self.localDepthDirectory + self.dailySummary)  
        figHourly.savefig(self.localDepthDirectory + self.hourlySummary)  

        self._cloudUpdate()
        plt.clf()
    
    def _summarizeBuilding(self, t0, t1, totalThreshold = None, minPixels = None, extremePixels = 0):  
        # Check times are good
        self._checkTimes(t0,t1)

        data = OrderedDict()
        # Get data
        data['projectID'] = self.lp.projectID
        bowerLocations = self._returnBowerLocations(t0, t1, cropped = True, totalThreshold = totalThreshold, minPixels = minPixels)
        heightChange = self._returnHeightChange(t0, t1, masked = True, cropped = True, totalThreshold = totalThreshold, minPixels = minPixels)
        data['castleArea'] = np.count_nonzero(bowerLocations == 1)*self.convertPixel*self.convertPixel
        data['pitArea'] = np.count_nonzero(bowerLocations == -1)*self.convertPixel*self.convertPixel
        data['totalArea'] = heightChange.shape[0]*heightChange.shape[1]*self.convertPixel*self.convertPixel
        data['castleVolume'] = np.nansum(heightChange[bowerLocations == 1])*self.convertPixel*self.convertPixel
        data['pitVolume'] = np.nansum(heightChange[bowerLocations == -1])*-1*self.convertPixel*self.convertPixel
        data['totalVolume'] = data['castleVolume'] + data['pitVolume']
        if extremePixels == 0:
            data['bowerIndex'] = (data['castleVolume'] - data['pitVolume'])/data['totalVolume']
        else:
            sortedData = np.sort(np.abs(heightChange[~np.isnan(heightChange)].flatten()))
            try:
                threshold = sortedData[-1*extremePixels]
            except IndexError:
                threshold = sortedData[0]
            numerator = np.nansum(heightChange[(bowerLocations == 1) & (heightChange > threshold)]) - -1*np.nansum(heightChange[(bowerLocations == -1) & (heightChange < -1*threshold)])
            denom = np.nansum(heightChange[(bowerLocations == 1) & (heightChange > threshold)]) + -1*np.nansum(heightChange[(bowerLocations == -1) & (heightChange < -1*threshold)])
            data['bowerIndex'] = numerator/denom
        return data

    def _returnBowerLocations(self, t0, t1, cropped = False, totalThreshold = None, minPixels = None):
        # Check times are good
        self._checkTimes(t0,t1)

        if cropped:
            self.loadTray()

        # Identify total height change and time change
        totalHeightChange = self._returnHeightChange(t0, t1, masked=False, cropped=False)
        timeChange = t1 - t0

        # Determine threshold and minimum size of bower to use based upon timeChange
        if timeChange.total_seconds() < 7300:
            # ~2 hours or less
            if totalThreshold is None:
                totalThreshold = 0.2
            if minPixels is None:
                minPixels = 1000
        elif timeChange.total_seconds() < 129600:
            # 2 hours to 1.5 days
            if totalThreshold is None:
                totalThreshold = 0.4
            if minPixels is None:
                minPixels = 1000
        else:
            #  1.5 days or more
            if totalThreshold is None:
                totalThreshold = 1.0
            if minPixels is None:
                minPixels = 1000

        tCastle = totalHeightChange.copy()
        tCastle[tCastle < totalThreshold] = 0
        tCastle[np.isnan(tCastle)] = 0
        tCastle[tCastle!=0] = 1
        tCastle = morphology.remove_small_objects(tCastle.astype(bool), minPixels)

        tPit = totalHeightChange.copy()
        tPit[tPit > -1*totalThreshold] = 0
        tPit[np.isnan(tPit)] = 0
        tPit[tPit!=0] = 1
        tPit = morphology.remove_small_objects(tPit.astype(bool), minPixels)

        totalHeightChange[tCastle == False] = 0
        totalHeightChange[tPit == False] = 0

        totalHeightChange[tCastle == True] = 1
        totalHeightChange[tPit == True] = -1

        if cropped:
            totalHeightChange = totalHeightChange[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]]

        return totalHeightChange
         
    def _returnHeightChange(self, t0, t1, masked = False, cropped = False, totalThreshold = None, minPixels = None):
        # Check times are good
        self._checkTimes(t0,t1)

        # Load necessary data
        self.loadSmoothedArray()
        if cropped:
            self.loadTray()
        
        # Find closest frames to desired times
        try:
            first_index = max([False if x.time<=t0 else True for x in self.lp.frames].index(True) - 1, 0) #This ensures that we get overnight changes when kinect wasn't running
        except ValueError:
            if t0 > self.lp.frames[-1].time:
                first_index = -1
            else:
                first_index = 0

        try:
            last_index = max([False if x.time<=t1 else True for x in self.lp.frames].index(True) - 1, 0)
        except ValueError:
            last_index = len(self.lp.frames) - 1
            
        change = self.smoothDepthData[first_index] - self.smoothDepthData[last_index]
        
        if masked:
            change[self._returnBowerLocations(t0, t1, totalThreshold = totalThreshold, minPixels = minPixels) == 0] = 0

        if cropped:
            change = change[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]]
            
        return change
      
    def _returnHeight(self, t, cropped = False):

        # Check times are good
        self._checkTimes(t)

        # Load necessary data
        self.loadSmoothedArray()
        if cropped:
            self.loadTray()

       # Find closest frames to desired times
        try:
            first_index = max([False if x.time<=t else True for x in self.lp.frames].index(True) - 1, 0) #This ensures that we get overnight changes when kinect wasn't running
        except ValueError:
            if t > self.lp.frames[-1].time:
                first_index = -1
            else:
                first_index = 0

        change = self.smoothDepthData[first_index]
        
        if cropped:
            change = change[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]]
            
        return change

    def _cloudUpdate(self):
        print('Syncing with cloud', file = sys.stderr)
        subprocess.call(['rclone', 'copy', self.localDepthDirectory, self.cloudDepthDirectory], stderr = self.fnull)
        #subprocess.call(['rclone', 'copy', self.localOutputDirectory, self.cloudOutputDirectory], stderr = self.fnull)

    def _checkTimes(self, t0, t1 = None):
        if t1 is None:
            if type(t0) != datetime.datetime:
                raise Exception('Timepoints to must be datetime.datetime objects')
            return
        # Make sure times are appropriate datetime objects
        if type(t0) != datetime.datetime or type(t1) != datetime.datetime:
            raise Exception('Timepoints to must be datetime.datetime objects')
        if t0 > t1:
            raise Exception('Second timepoint must be greater than first timepoint')

    def _print(self, outtext):
        print(str(getpass.getuser()) + ' analyzed at ' + str(datetime.datetime.now()) + ' on ' + socket.gethostname() + ': ' + outtext, file = self.anLF)
        print(outtext, file = sys.stderr)
        self.anLF.close() # Close and reopen file to flush it
        self.anLF = open(self.localDepthDirectory + 'DepthAnalysisLog.txt', 'a')

