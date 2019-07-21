import subprocess, os, sys, datetime, shutil, pdb
import numpy as np
import pandas as pd

from Modules.Analysis.DepthProcessor import DepthProcessor as DP
from Modules.LogParser import LogParser as LP
from Modules.Analysis.VideoProcessor import VideoProcessor as VP


class DataAnalyzer:
    def __init__(self, projectID, remote, locDir, cloudDir, rewriteFlag):
        
        # Store input
        self.projectID = projectID
        self.remote = remote
        self.localMasterDirectory = locDir + projectID + '/'
        self.cloudMasterDirectory = remote + ':' + cloudDir + projectID + '/'
        self.transMFile = 'depthVideoTransformation.npy'
        self.transFig = 'depthVideoTransformation.pdf'
        
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
        #print('Beginning analysis of: ' + self.projectID + ' taken from tank: ' + self.lp.tankID, file = sys.stderr)
        #print(str(len(self.lp.frames)) + ' total frames for this project from ' + str(self.lp.frames[0].time) + ' to ' + str(self.lp.frames[-1].time), file = sys.stderr)
        #print(str(len(self.lp.movies)) + ' total videos for this project.', file = sys.stderr)
        if self.rewriteFlag:
            print('Requested data will be reanalyzed from start to finish', file = sys.stderr)
        
        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')

        # Create Depth object (low overhead even if video is just processed)
        self.depthObj = DP(self.localMasterDirectory, self.cloudMasterDirectory, self.logfile)
   
    def __del__(self):
        # Remove local files once object is destroyed
        #shutil.rmtree(self.localMasterDirectory)
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False
        
    def prepareData(self):
        print('Preparing ' + self.projectID, file = sys.stderr)
        if self.rewriteFlag:
            self.depthObj.createTray()
        else:
            self.depthObj.loadTray()
        
    def processDepth(self):
        print('Processing ' + self.projectID, file = sys.stderr)

        self.depthObj.loadTray()
        if self.rewriteFlag:
            self.depthObj.createSmoothedArray()
            self.depthObj.createBowerLocations()
        else:
            self.depthObj.loadSmoothedArray()

        self.depthObj.createDataSummary()
        
    def processVideos(self, index, rewriteClusters, rewriteSummaries):

        self._loadRegistration()

        # Create Video objects (low overhead even if video is not processed)
        for x in index:
            vo = VP(self.projectID, self.lp.movies[x-1], self.localMasterDirectory, self.cloudMasterDirectory, self.transM, self.depthObj)
            if self.rewriteFlag:
                print('Rewriting all video data for ' + self.projectID + ' and videos ' + str(index), file = sys.stderr)
                vo.loadVideo()
                vo.createHMM()
                vo.createClusterSummary(j)
                vo.createClusterClips()
                #vo.summarizeData()
                vo.cleanup()
            elif rewriteClusters:
                print('Rewriting cluster data for ' + self.projectID + ' and videos ' + str(index), file = sys.stderr)
                vo.createClusters()
                vo.createClusterSummary()
                vo.createClusterClips()
                vo.cleanup()

            elif rewriteSummaries:
                print('Rewriting cluster summary and clips for ' + self.projectID + ' and videos ' + str(index), file = sys.stderr)
                vo.createClusterSummary()
                vo.createClusterClips()
                vo.cleanup()

            else:
                print('Rewriting cluster clips for ' + self.projectID + ' and videos ' + str(index), file = sys.stderr)
                vo.createClusterClips()
                vo.cleanup()

    def fixIssues(self, index, mlDirectory):
        self._loadRegistration()

        for x in index:
            vo = VP(self.projectID, self.lp.movies[x-1], self.localMasterDirectory, self.cloudMasterDirectory, self.transM, self.depthObj)
            vo._fixData(mlDirectory)
            vo.cleanup()

        # Create Video objects (low overhead even if video is not processed)
    
            
    def labelVideos(self, index, mainDT, cloudMLDirectory, number):
        self._loadRegistration()

        # Create Video objects (low overhead even if video is not processed)            
        for x in index:
            vo = VP(self.projectID, self.lp.movies[x-1], self.localMasterDirectory, self.cloudMasterDirectory, self.transM, self.depthObj)
            vo.labelClusters(self.rewriteFlag, mainDT, cloudMLDirectory, number)

    def countFish(self, index, cloudCountDirectory):
        self._loadRegistration()
        self.videoObjs = [VP(self.projectID, x, self.localMasterDirectory, self.cloudMasterDirectory, self.transM) for x in self.lp.movies]
        if index is None:
            vos = self.videoObjs
        else:
            vos = [self.videoObjs[x-1] for x in index]
            
        for vo in vos:
            vo.countFish(self.rewriteFlag, cloudCountDirectory)
            break


    def predictLabels(self, index, modelLocation, modelIDs):
        self._loadRegistration()

        print(modelLocation)
            
        clusterData = []
        for x in index:
            vo = VP(self.projectID, self.lp.movies[x-1], self.localMasterDirectory, self.cloudMasterDirectory, self.transM, self.depthObj)
            clusterData.append(vo.predictLabels(modelLocation, modelIDs))

        fullClusterData = pd.concat(clusterData, ignore_index=True)

        fullClusterData.to_csv(self.localMasterDirectory + 'VideoAnalysis/AllClusterData.csv', sep = ',')
        subprocess.call(['rclone', 'copy', self.localMasterDirectory + 'VideoAnalysis/AllClusterData.csv', self.cloudMasterDirectory + 'VideoAnalysis/'])


    def summarizeData(self):
        pass
    
    def cleanup(self):
        shutil.rmtree(self.localMasterDirectory)

    def _loadRegistration(self):
        try:
            self.transM
            return
        except AttributeError:
            pass

        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.transMFile, self.localMasterDirectory], stderr = self.fnull)
        if os.path.isfile(self.localMasterDirectory + self.transMFile):
            print('Loading transformation matrix information from file on dropbox')
            self.transM = np.load(self.localMasterDirectory + self.transMFile)
            return

        else:
            self._createRegistration()
        
    def _createRegistration(self):

        import cv2
        import matplotlib.pyplot as plt
        from Modules.Analysis.roipoly import roipoly

        # Unable to load it from existing file, either because it doesn't exist or the rewrite flag was set
        print('Registering RGB and Depth data ')
        # Find first videofile during the day
        videoObj = [x for x in self.lp.movies if x.time.hour >= 8 and x.time.hour <= 20][0]
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + videoObj.pic_file, self.localMasterDirectory + videoObj.movieDir], stderr = self.fnull)
        
        # Find depthfile that is closest to the video file time
        depthObj = [x for x in self.lp.frames if x.time > videoObj.time][0]
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + depthObj.pic_file, self.localMasterDirectory + depthObj.frameDir], stderr = self.fnull)
                        
        if not os.path.isfile(self.localMasterDirectory + videoObj.pic_file) or not os.path.isfile(self.localMasterDirectory + depthObj.pic_file):
            print('Cant find RGB pictures of both Kinect and PiCamera')
            self.transM = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype = 'float64')
            np.save(self.localMasterDirectory + self.transMFile, self.transM)
            subprocess.call(['rclone', 'copy', self.localMasterDirectory + self.transMFile, self.cloudMasterDirectory], stderr = self.fnull)
            return

        im1 =  cv2.imread(self.localMasterDirectory + depthObj.pic_file)
        im2 =  cv2.imread(self.localMasterDirectory + videoObj.pic_file)
        im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

        while True:
            fig = plt.figure(figsize=(18, 12))
            ax1 = fig.add_subplot(1,2,1)       
            ax2 = fig.add_subplot(1,2,2)
        
            ax1.imshow(im1_gray, cmap='gray')
            ax2.imshow(im2_gray, cmap='gray')

            ax1.set_title('Select four points in this object (Double-click on the fourth point)')
            ROI1 = roipoly(roicolor='r')
            plt.show()
            fig = plt.figure(figsize=(18, 12))
            ax1 = fig.add_subplot(1,2,1)       
            ax2 = fig.add_subplot(1,2,2)
  
            ax1.imshow(im1_gray, cmap='gray')
            ROI1.displayROI(ax = ax1)
            ax2.imshow(im2_gray, cmap='gray')

            ax2.set_title('Select four points in this object (Double-click on the fourth point)')
            ROI2 = roipoly(roicolor='b')
            plt.show()

            ref_points =[[ROI1.allxpoints[0], ROI1.allypoints[0]], [ROI1.allxpoints[1], ROI1.allypoints[1]], [ROI1.allxpoints[2], ROI1.allypoints[2]], [ROI1.allxpoints[3], ROI1.allypoints[3]]]
            new_points =[[ROI2.allxpoints[0], ROI2.allypoints[0]], [ROI2.allxpoints[1], ROI2.allypoints[1]], [ROI2.allxpoints[2], ROI2.allypoints[2]], [ROI2.allxpoints[3], ROI2.allypoints[3]]]

            if len(ROI1.allxpoints) != 4 or len(ROI2.allxpoints) != 4:
                print('Wrong length, ROI1 = ' + str(len(ROI1.allxpoints)) + ', ROI2 = ' + str(len(ROI2.allxpoints)))
                continue
        
            self.transM = cv2.getPerspectiveTransform(np.float32(new_points),np.float32(ref_points))
            newImage = cv2.warpPerspective(im2_gray, self.transM, (640, 480))

            fig = plt.figure(figsize=(18, 12))
            ax1 = fig.add_subplot(1,2,1)       
            ax2 = fig.add_subplot(1,2,2)
        
            ax1.imshow(im1_gray, cmap='gray')
            ax2.imshow(newImage, cmap='gray')
            
            fig.savefig(self.localMasterDirectory + self.transFig)

            plt.show()

            userInput = input('Type q if this is acceptable: ')
            if userInput == 'q':
                break


        np.save(self.localMasterDirectory + self.transMFile, self.transM)
        subprocess.call(['rclone', 'copy', self.localMasterDirectory + self.transMFile, self.cloudMasterDirectory], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.localMasterDirectory + self.transFig, self.cloudMasterDirectory], stderr = self.fnull)

                


