import os, sys, psutil, subprocess, pims, datetime, shutil, cv2, math, getpass, socket, random
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pandas as pd
from hmmlearn import hmm
from Modules.Analysis.HMM_data import HMMdata
from Modules.Analysis.MachineLabel import MachineLabelAnalyzer as MLA

from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage

#import pims, math, psutil, shutil, os, datetime, subprocess, sys, pickle, cv2
#import scipy.ndimage
#import matplotlib.pyplot as plt
#from multiprocessing.dummy import Pool as ThreadPool
#from Modules.HMM_data import HMMdata
#from PIL import Image
#from sklearn.cluster import DBSCAN
#from sklearn.neighbors import radius_neighbors_graph
#from sklearn.neighbors import NearestNeighbors
np.warnings.filterwarnings('ignore')


class VideoProcessor:
    # This class takes in an mp4 videofile and an output directory and performs the following analysis on it:
    # 1. Performs HMM analysis on all pixel files
    # 3. Clusters HMM data to identify potential spits and scoops
    # 4. Uses DeepLabCut to annotate fish

    #Parameters - blocksize, 
    def __init__(self, projectID, videoObj, localMasterDirectory, cloudMasterDirectory, transM):

        # Main location for machine learning clips
        
        # Store arguments
        self.projectID = projectID
        self.videofile = videoObj.mp4_file
        self.height = videoObj.height
        self.width = videoObj.width
        self.frame_rate = videoObj.framerate
        self.startTime = videoObj.time

        self.localMasterDirectory = localMasterDirectory if localMasterDirectory[-1] == '/' else localMasterDirectory + '/'
        self.cloudMasterDirectory = cloudMasterDirectory if cloudMasterDirectory[-1] == '/' else cloudMasterDirectory + '/'
        self.transM = transM

        self.baseName = self.videofile.split('/')[-1].split('.')[0]
        self.localVideoDirectory = self.localMasterDirectory + 'VideoAnalysis/' + self.baseName + '/'
        self.cloudVideoDirectory = self.cloudMasterDirectory + 'VideoAnalysis/' + self.baseName + '/'
        
        self.localClusterDirectory = self.localVideoDirectory + 'ClusterData/'
        self.cloudClusterDirectory = self.cloudVideoDirectory + 'ClusterData/'

        self.tempDirectory = self.localVideoDirectory + 'Temp/'

        self.localManualLabelClipsDirectory = self.localClusterDirectory + 'ManualLabelClips/'
        self.cloudManualLabelClipsDirectory = self.cloudClusterDirectory + 'ManualLabelClips/'

        self.localAllClipsDirectory = self.localClusterDirectory + 'AllClips/'
        self.cloudAllClipsDirectory = self.cloudClusterDirectory + 'AllClips/'

        os.makedirs(self.localManualLabelClipsDirectory) if not os.path.exists(self.localManualLabelClipsDirectory) else None
        os.makedirs(self.localAllClipsDirectory) if not os.path.exists(self.localAllClipsDirectory) else None

        # Set paramaters
        self.cores = psutil.cpu_count() # Number of cores that should be used to analyze the video

        # Create file names
        self.hmmFile = self.baseName + '.hmm.npy'
        self.clusterFile = 'LabeledClusters.csv'
        self.labeledCoordsFile = 'LabeledCoords.npy'

        #print('VideoProcessor: Analyzing ' + self.videofile, file = sys.stderr)

        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')

        self.anLF = open(self.localVideoDirectory + 'VideoAnalysisLog.txt', 'a')

    def __del__(self):
        pass
        #subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'VideoAnalysisLog.txt', self.cloudVideoDirectory])
        #shutil.rmtree(self.localVideoDirectory)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False        
    
    def loadVideo(self):
        if os.path.isfile(self.localMasterDirectory + self.videofile):
            print(self.videofile + ' present in local path.', file = sys.stderr)
        else:
            print(self.videofile + ' not present in local path. Trying to find it remotely', file = sys.stderr)
            # Try to download it from the cloud
            subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.videofile, self.localMasterDirectory + self.videofile.split(self.videofile.split('/')[-1])[0]], stderr = self.fnull)                
            if not os.path.isfile(self.localMasterDirectory + self.videofile):
                print(self.videofile + ' not present in remote path. Trying to find h264 file and convert it to mp4', file = sys.stderr)
                if not os.path.isfile(self.localMasterDirectory + self.videofile.replace('.mp4', '.h264')):
                    subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.videofile.replace('.mp4', '.h264'), self.localMasterDirectory + self.videofile.split(self.videofile.split('/')[-1])[0]], stderr = self.fnull)                

                if not os.path.isfile(self.localMasterDirectory + self.videofile.replace('.mp4', '.h264')):
                    print('Unable to find ' + self.videofile.replace('.mp4', '.h264'), file = sys.stderr)
                    raise Exception

                # Convert it to mpeg
                subprocess.call(['ffmpeg', '-i', self.localMasterDirectory + self.videofile.replace('.mp4', '.h264'), '-c:v', 'copy', self.localMasterDirectory + self.videofile])
                
                if os.stat(self.localMasterDirectory + self.videofile).st_size >= os.stat(self.localMasterDirectory + self.videofile.replace('.mp4', '.h264')).st_size:
                    try:
                        vid = pims.Video(self.localMasterDirectory + self.videofile)
                        vid.close()
                        os.remove(self.localMasterDirectory + self.videofile.replace('.mp4', '.h264'))
                    except Exception as e:
                        self._print(e)
                        self._print('Unable to convert ' + self.videofile)
                        raise Exception
                    print('Uploading ' + self.videofile + ' to cloud', file = sys.stderr)
                    subprocess.call(['rclone', 'copy', self.localMasterDirectory + self.videofile, self.cloudMasterDirectory + self.videofile.split(self.videofile.split('/')[-1])[0]], stderr = self.fnull)
                self._print(self.videofile + ' converted and uploaded to cloud')

        #Grab info on video
        cap = pims.Video(self.localMasterDirectory + self.videofile)
        self.height = int(cap.frame_shape[0])
        self.width = int(cap.frame_shape[1])
        self.frame_rate = int(cap.frame_rate)
        try:
            self.frames = min(int(cap.get_metadata()['duration']*cap.frame_rate), 12*60*60*self.frame_rate)
        except AttributeError:
            self.frames = min(int(cap.duration*cap.frame_rate), 12*60*60*self.frame_rate)
        cap.close()

    def loadHMM(self):
        try:
            self.obj
        except AttributeError:
            if not os.path.isfile(self.cloudVideoDirectory + self.hmmFile):
                subprocess.call(['rclone', 'copy', self.cloudVideoDirectory + self.hmmFile, self.localVideoDirectory], stderr = self.fnull)
                subprocess.call(['rclone', 'copy', self.cloudVideoDirectory + self.hmmFile.replace('.npy', '.txt'), self.localVideoDirectory], stderr = self.fnull)

            if not os.path.isfile(self.localVideoDirectory + self.hmmFile):
                self.createHMM()
                return
            else:
                self.obj = HMMdata(filename = self.localVideoDirectory + self.hmmFile)

    def loadClusters(self):
        try:
            self.labeledCoords
        except AttributeError:
            if not os.path.isfile(self.localClusterDirectory + self.labeledCoordsFile):
                subprocess.call(['rclone', 'copy', self.cloudClusterDirectory + self.labeledCoordsFile, self.localClusterDirectory], stderr = self.fnull)
                
            if not os.path.isfile(self.localClusterDirectory + self.labeledCoordsFile):
                self.createClusters()
                return
            else:
                self.labeledCoords = np.load(self.localClusterDirectory + self.labeledCoordsFile)
          
    def loadClusterSummary(self):
        try:
            self.clusterData
        except AttributeError:
            if not os.path.isfile(self.localClusterDirectory + self.clusterFile):
                subprocess.call(['rclone', 'copy', self.cloudClusterDirectory + self.clusterFile, self.localClusterDirectory], stderr = self.fnull)
                
            if not os.path.isfile(self.localClusterDirectory + self.clusterFile):
                self.createClusterSummary()
                return
            else:
                self.clusterData = pd.read_csv(self.localClusterDirectory + self.clusterFile, sep = ',', header = 0, index_col = 0)
                                  
    def createHMM(self, blocksize = 5*60, window = 120, hmm_time = 60*60):
        """
        This functon decompresses video into smaller chunks of data formated in the numpy array format.
        Each numpy array contains one row of data for the entire video.
        This function then smoothes the raw data
        Finally, an HMM is fit to the data and an HMMobject is created
        """
        
        #Download video
        self.loadVideo()

        os.makedirs(self.tempDirectory) if not os.path.exists(self.tempDirectory) else None
        
        self.blocksize = blocksize
        self.window = window
        self.hmm_time = hmm_time
        
        total_blocks = math.ceil(self.frames/(blocksize*self.frame_rate)) #Number of blocks that need to be analyzed for the full video

        # Step 1: Convert mp4 to npy files for each row
        pool = ThreadPool(self.cores) #Create pool of threads for parallel analysis of data
        start = datetime.datetime.now()
        self._print('Created ' + self.hmmFile)
        print('TotalBlocks: ' + str(total_blocks), file = sys.stderr)
        print('TotalThreads: ' + str(self.cores), file = sys.stderr)
        print('Video processed: ' + str(self.blocksize/60) + ' min per block, ' + str(self.blocksize/60*self.cores) + ' min per cycle', file = sys.stderr)
        print('Converting mp4 data to npy arrays at 1 fps', file = sys.stderr)
        print('StartTime: ' + str(start), file = sys.stderr)
        
        for i in range(0, math.ceil(total_blocks/self.cores)):
            blocks = list(range(i*self.cores, min(i*self.cores + self.cores, total_blocks)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ', Processing blocks: ' + str(blocks[0]) + ' to ' +  str(blocks[-1]))
            results = pool.map(self._readBlock, blocks)
            print('Data read: ' + str((datetime.datetime.now() - start).seconds) + ' seconds')
            for row in range(self.height):
                row_file = self._row_fn(row)
                out_data = np.concatenate([results[x][row] for x in range(len(results))], axis = 1)
                if os.path.isfile(row_file):
                    out_data = np.concatenate([np.load(row_file),out_data], axis = 1)
                np.save(row_file, out_data)
            print('Data wrote: ' + str((datetime.datetime.now() - start).seconds) + ' seconds', file = sys.stderr)
        pool.close() 
        pool.join() 
        print('TotalTime: ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes', file = sys.stderr)

        # Step 2: Smooth data to remove outliers
        pool = ThreadPool(self.cores)
        start = datetime.datetime.now()
        print('Smoothing data to filter out outliers', file = sys.stderr)
        print('StartTime: ' + str(start), file = sys.stderr)
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ' seconds, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]), file = sys.stderr)
            results = pool.map(self._smoothRow, rows)
        print('TotalTime: Took ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes to smooth ' + str(self.height) + ' rows')
        pool.close() 
        pool.join()

        # Step 3: Calculate HMM values for each row
        pool = ThreadPool(self.cores)
        start = datetime.datetime.now()
        print('Calculating HMMs for all data', file = sys.stderr)
        print('StartTime: ' + str(start), file = sys.stderr)
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            print('Hours since start: ' + str((datetime.datetime.now() - start).seconds/3600) + ' hours, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]), file = sys.stderr)
            results = pool.map(self._hmmRow, rows)
        print('TotalTime: Took ' + str((datetime.datetime.now() - start).seconds/3600) + ' hours to calculate HMMs for ' + str(self.height) + ' rows', file = sys.stderr)
        pool.close() 
        pool.join()
        
        # Step 4: Create HMM object and delete temporary data if necessary
        start = datetime.datetime.now()
        print('Converting HMMs to internal data structure and keeping temporary data', file = sys.stderr)

        print('StartTime: ' + str(start), file = sys.stderr)
        
        self.obj = HMMdata(self.width, self.height, self.frames, self.frame_rate)
        self.obj.add_data(self.tempDirectory, self.localVideoDirectory + self.hmmFile)
        # Copy example data to directory containing videofile
        subprocess.call(['cp', self._row_fn(int(self.height/2)), self._row_fn(int(self.height/2)).replace('.npy', '.smoothed.npy'), self._row_fn(int(self.height/2)).replace('.npy', '.hmm.npy'), self.localVideoDirectory])

        shutil.rmtree(self.tempDirectory)
      
        subprocess.call(['rclone', 'copy', self.localVideoDirectory, self.cloudVideoDirectory], stderr = self.fnull)

        print('Took ' + str((datetime.datetime.now() - start).seconds/60) + ' convert HMMs', file = sys.stderr)

    def createClusters(self, minMagnitude = 0, treeR = 22, leafNum = 190, neighborR = 22, timeScale = 10, eps = 18, minPts = 90, delta = 1.0):
        #self.loadVideo()
        self.loadHMM()
        self._print('Created ' + self.labeledCoordsFile)
        coords = self.obj.retDBScanMatrix(minMagnitude)
        np.save(self.localClusterDirectory + 'RawCoords.npy', coords)
        #subprocess.call(['rclone', 'copy', self.localClusterDirectory + 'RawCoordsFile.npy', self.cloudClusterDirectory], stderr = self.fnull)
               

        sortData = coords[coords[:,0].argsort()][:,0:3] #sort data by time for batch processing, throwing out 4th column (magnitude)
        numBatches = int(sortData[-1,0]/delta/3600) + 1 #delta is number of hours to batch together. Can be fraction.

        sortData[:,0] = sortData[:,0]*timeScale #scale time so that time distances between transitions are comparable to spatial differences
        labels = np.zeros(shape = (sortData.shape[0],1), dtype = sortData.dtype)

        #Calculate clusters in batches to avoid RAM overuse
        curr_label = 0 #Labels for each batch start from zero - need to offset these
            
        print('Calculating clusters in ' + str(numBatches) + ' total batches', file = sys.stderr)
        for i in range(numBatches):
            print('Batch: ' + str(i), file = sys.stderr)
            min_time, max_time = i*delta*timeScale*3600, (i+1)*delta*timeScale*3600 # Have to deal with rescaling of time. 3600 = # seconds in an hour
            hour_range = np.where((sortData[:,0] > min_time) & (sortData[:,0] <= max_time))
            min_index, max_index = hour_range[0][0], hour_range[0][-1] + 1
            X = NearestNeighbors(radius=treeR, metric='minkowski', p=2, algorithm='kd_tree',leaf_size=leafNum,n_jobs=24).fit(sortData[min_index:max_index])
            dist = X.radius_neighbors_graph(sortData[min_index:max_index], neighborR, 'distance')
            sub_label = DBSCAN(eps=eps, min_samples=minPts, metric='precomputed', n_jobs=24).fit_predict(dist)
            new_labels = int(sub_label.max()) + 1
            sub_label[sub_label != -1] += curr_label
            labels[min_index:max_index,0] = sub_label
            curr_label += new_labels

        sortData[:,0] = sortData[:,0]/timeScale
        self.labeledCoords = np.concatenate((sortData, labels), axis = 1).astype('int64')
        np.save(self.localClusterDirectory + self.labeledCoordsFile, self.labeledCoords)
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.labeledCoordsFile, self.cloudClusterDirectory], stderr = self.fnull)

    def createClusterSummary(self, Nclips = 200):
        #self.loadVideo()
        self.loadHMM()
        self.loadClusters()
        self._print('Created ' + self.clusterFile)

        uniqueLabels = set(self.labeledCoords[:,3])
        uniqueLabels.remove(-1)
        print(self.projectID + '\t' + self.baseName + ': ' + str(self.labeledCoords[self.labeledCoords[:,3] != -1].shape[0]) + ' HMM transitions assigned to ' + str(len(uniqueLabels)) + ' clusters', file = sys.stderr)
        print('Grouping data', file = sys.stderr)

        df = pd.DataFrame(self.labeledCoords, columns=['T','X','Y','LID'])
        clusterData = df.groupby('LID').apply(lambda x: pd.Series({
            'projectID': self.projectID,
            'videoID': self.baseName,
            'N': x['T'].count(),
            't': int(x['T'].mean()),
            'X': int(x['X'].mean()),
            'Y': int(x['Y'].mean()),
            't_span': int(x['T'].max() - x['T'].min()),
            'X_span': int(x['X'].max() - x['X'].min()),
            'Y_span': int(x['Y'].max() - x['Y'].min()),
            'ManualAnnotation': 'No',
            'ManualLabel': '',
            'ClipCreated': '',
        })
        )
        print('Calculating new coordinates', file = sys.stderr)

        clusterData['X_depth'] = clusterData.apply(lambda row: (self.transM[0][0]*row.X + self.transM[0][1]*row.Y + self.transM[0][2])/(self.transM[2][0]*row.X + self.transM[2][1]*row.Y + self.transM[2][2]), axis=1)
        clusterData['Y_depth'] = clusterData.apply(lambda row: (self.transM[1][0]*row.X + self.transM[1][1]*row.Y + self.transM[1][2])/(self.transM[2][0]*row.X + self.transM[2][1]*row.Y + self.transM[2][2]), axis=1)
        clusterData['TimeStamp'] = clusterData.apply(lambda row: (self.startTime + datetime.timedelta(seconds = int(row.t))), axis=1)

        clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        self.clusterData = pd.read_csv(self.localClusterDirectory + self.clusterFile, sep = ',', header = 0)
        
        # Identify rows for manual labeling
        self._identifyManualClusters(Nclips)
        
    def _identifyManualClusters(self, Nclips = 200, delta_xy = 100, delta_t = 60, smallLimit = 500):
    
        self.loadClusterSummary()

        clipsCreated = self.clusterData.groupby('ManualAnnotation').count()['LID']['Yes']
        print(clipsCreated)

        # Identify rows for manual labeling
        smallClips = 0

        print('Identifying clusters for manual annotation', file = sys.stderr)

        for row in self.clusterData.sample(n = self.clusterData.shape[0]).itertuples():
            if clipsCreated > Nclips:
                break
            
            LID, N, t, x, y = row.LID, row.N, row.t, row.X, row.Y
            try:
                manualAnnotation = row.ManualAnnotation
            except AttributeError:
                self.clusterData['ManualAnnotation'] = 'No'
                manualAnnotation = 'No'
            if manualAnnotation == 'Yes':
                continue
            try:
                manualLabel = row.ManualLabel
                if manualLabel is not np.nan and manualLabel != '':
                    continue
            except AttributeError:
                continue
            if x - delta_xy < 0 or x + delta_xy >= self.height or y - delta_xy < 0 or y + delta_xy >= self.width or LID == -1 or self.frame_rate*t - delta_t <0:
                continue
            if N < smallLimit:
                if smallClips > Nclips/20:
                    continue
            self.clusterData.loc[self.clusterData.LID == LID,'ManualAnnotation'] = 'Yes'
            clipsCreated += 1
            if N < smallLimit:
                smallClips += 1

        
        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)

    def createClusterClips(self, delta_xy = 100, delta_t = 60, smallLimit = 500, manualOnly = False):
        print('Creating clip files for machine learning', file = sys.stderr)
        self.loadVideo()
        self.loadHMM()
        self.loadClusters()
        self.loadClusterSummary()
        self._print('Creating manual label clip videos, and clip videos for all clusters')

        cap = cv2.VideoCapture(self.localMasterDirectory + self.videofile)
        count = 0
        for row in self.clusterData.itertuples():
            #if count ==30:
            #    break
            LID, N, t, x, y, ml = row.LID, row.N, row.t, row.X, row.Y, row.ManualAnnotation
            if x - delta_xy < 0 or x + delta_xy >= self.height or y - delta_xy < 0 or y + delta_xy >= self.width or LID == -1 or self.frame_rate*t - delta_t <0 or self.frame_rate*t+delta_t >= self.frames:
                #print('Cannot create clip for: ' + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y), file = sys.stderr)
                self.clusterData.loc[self.clusterData.LID == LID,'ClipCreated'] = 'No'
                continue
            #print('ffmpeg')
            #command = ['ffmpeg', '-i', self.localMasterDirectory + self.videofile, '-filter:v', 'crop=' + str(2*delta_xy) + ':' + str(2*delta_xy) + ':' + str(y-delta_xy) + ':' + str(x-delta_xy) + '', '-ss', str(t - int(delta_t/self.frame_rate)), '-frames:v', str(2*delta_t), self.localAllClipsDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4']
            #t1 = datetime.datetime.now()
            #subprocess.call(command, stderr = self.fnull)
            #t2 = datetime.datetime.now()
            #try:
            #    ffmpegTime += t2-t1
            #except:
            #    ffmpegTime = t2 - t1

            if not manualOnly:
                outAll = cv2.VideoWriter(self.localAllClipsDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), self.frame_rate, (2*delta_xy, 2*delta_xy))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.frame_rate*(t) - delta_t))
                for i in range(delta_t*2):
                    ret, frame = cap.read()
                    outAll.write(frame[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy])
                outAll.release()
                count+=1
            #t3 = datetime.datetime.now()
            #try:
            #    cvTime += t3-t2
            #except:
            #    cvTime = t3 - t2
            #print('ff: ' + str(ffmpegTime) + ' cv: ' + str(cvTime))
                self.clusterData.loc[self.clusterData.LID == LID,'ClipCreated'] = 'Yes'

            if ml == 'Yes':
                outAllHMM = cv2.VideoWriter(self.localManualLabelClipsDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '_ManualLabel.mp4', cv2.VideoWriter_fourcc(*"mp4v"), self.frame_rate, (4*delta_xy, 2*delta_xy))

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.frame_rate*(t) - delta_t))
                HMMChanges = self.obj.ret_difference(self.frame_rate*(t) - delta_t, self.frame_rate*(t) + delta_t)
                clusteredPoints = self.labeledCoords[self.labeledCoords[:,3] == LID][:,1:3]

                for i in range(delta_t*2):

                    ret, frame = cap.read()
                    frame2 = frame.copy()
                    frame[HMMChanges != 0] = [300,125,125]
                    for coord in clusteredPoints: # This can probably be improved to speed up clip generation (get rid of the python loop)
                        frame[coord[0], coord[1]] = [125,125,300]
                    if ret:
                        outAllHMM.write(np.concatenate((frame2[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy], frame[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy]), axis = 1))

                outAllHMM.release()

        print('Syncing data', file = sys.stderr)
        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        subprocess.call(['rclone', 'delete', self.cloudManualLabelClipsDirectory], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.localManualLabelClipsDirectory, self.cloudManualLabelClipsDirectory], stderr = self.fnull)
        if not manualOnly:
            subprocess.call(['rclone', 'delete', self.cloudAllClipsDirectory], stderr = self.fnull)
            subprocess.call(['rclone', 'copy', self.localAllClipsDirectory, self.cloudAllClipsDirectory], stderr = self.fnull)


    def loadClusterClips(self):
        subprocess.call(['rclone', 'copy', self.cloudAllClipsDirectory, self.localAllClipsDirectory], stderr = self.fnull)
   
    def labelClusters(self, rewrite, mainDT, cloudMLDirectory):

        self._print('Labeling cluster')
        self.loadClusterSummary()
        subprocess.call(['rclone', 'copy', self.cloudManualLabelClipsDirectory, self.localManualLabelClipsDirectory], stderr = self.fnull)

        if 'MLabeler' not in self.clusterData:
            self.clusterData['MLabeler'] = ''

        if 'MLabelTime' not in self.clusterData:
            self.clusterData['MLabelTime'] = ''
            
        if rewrite:
            self.clusterData['ManualLabel'] = ''
            self.clusterData['MLabelTime'] = ''
            self.clusterData['MLabeler'] = ''
            self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
            subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)

        clips = [x for x in os.listdir(self.localManualLabelClipsDirectory) if '.mp4' in x]

        categories = ['c','f','p','t','b','m','s','x','o','d','q']

        print("Type 'c': build scoop; 'f': feed scoop; 'p': build spit; 't': feed spit; 'b': build multiple; 'm': feed multiple; 'd': drop sand; s': spawn; 'o': fish other; 'x': nofish other; 'q': quit")
        
        newClips = []
        for f in clips:
            clusterID = int(f.split('_')[0])

            # If already labeled and rewrite = False, then skip
            if not rewrite and not (self.clusterData.loc[self.clusterData.LID == clusterID,'ManualLabel'].values[0] is np.nan or self.clusterData.loc[self.clusterData.LID == clusterID,'ManualLabel'].values[0] == ''):
                continue
            
            cap = cv2.VideoCapture(self.localManualLabelClipsDirectory + f)
            
            while(True):

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cv2.imshow("LID: " + str(clusterID) + "; Type 'c': build scoop; 'f': feed scoop; 'p': build spit; 't': feed spit; 'b': build multiple; 'm': feed multiple; 'd': drop sand; s': spawn; 'o': fish other; 'x': nofish other; 'q': quit",cv2.resize(frame,(0,0),fx=4, fy=4))
                info = cv2.waitKey(25)
            
                if info in [ord(x) for x in categories]:
                    for i in range(1,10):
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                    break

            if info == ord('q'):
                break
            
            self.clusterData.loc[self.clusterData.LID == clusterID, 'ManualLabel'] = chr(info)
            self.clusterData.loc[self.clusterData.LID == clusterID, 'MLabeler'] = socket.gethostname()
            self.clusterData.loc[self.clusterData.LID == clusterID, 'MLabelTime'] = str(datetime.datetime.now())
            
            newClips.append(f.replace('_ManualLabel',''))

        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)

        print('Updating ML directories...')
        for clip in newClips:
            print(['rclone', 'copy', self.cloudAllClipsDirectory + clip, cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName ])
            subprocess.call(['rclone', 'copy', self.cloudAllClipsDirectory + clip, cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName])
            
        subprocess.call(['rclone', 'copy', cloudMLDirectory + mainDT, self.localClusterDirectory], stderr = self.fnull)
        tempData = pd.read_csv(self.localClusterDirectory + mainDT, sep = ',', header = 0, index_col = 0)
        tempData2 = pd.concat([tempData, self.clusterData[self.clusterData.ManualLabel != ''].dropna(subset=['ManualLabel'])], sort = False)
        
        tempData2.drop_duplicates(subset=['projectID', 'videoID', 'LID'], inplace=True, keep='last')

        tempData2.to_csv(self.localClusterDirectory + mainDT, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + mainDT, cloudMLDirectory], stderr = self.fnull)

    def predictLabels(self, modelLocation):
        print('Loading Cluster file')
        self.loadClusters()
        print('Loading Cluster files')
        self.loadClusterClips()
        print('Copying model data')
        subprocess.call(['rclone', 'copy', modelLocation + 'classInd.txt', self.localAllClipsDirectory], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', modelLocation + 'model.pth', self.localAllClipsDirectory], stderr = self.fnull)
        MLobj = MLA(self.projectID, self.baseName, self.localAllClipsDirectory, self.clusterFile)
        MLobj.prepareData()
        MLobj.makePredictions()

    def summarizeData(self):
        self.loadClusters()
        pass
        t_hours = int(self.frames/(self.frame_rate*60*60))
        rel_diff = np.zeros(shape = (t_hours, self.height, self.width), dtype = 'uint8')
        
        for i in range(1,t_hours+1):
            rel_diff[i-1] = self.obj.ret_difference((i-1)*60*60*25,i*60*60*25 - 1)

        return rel_diff

    def cleanup(self):
        shutil.rmtree(self.localVideoDirectory)
        if os.path.exists(self.localMasterDirectory + self.videofile):
            os.remove(self.localMasterDirectory + self.videofile)
        subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'VideoAnalysisLog.txt', self.cloudVideoDirectory])
            
    def _readBlock(self, block):
        min_t = block*self.blocksize
        max_t = min((block+1)*self.blocksize, int(self.frames/self.frame_rate))
        ad = np.empty(shape = (self.height, self.width, max_t - min_t), dtype = 'uint8')
        cap = pims.Video(self.localMasterDirectory + self.videofile)
        counter = 0
        for i in range(min_t, max_t):
            current_frame = i*self.frame_rate
            frame = cap[current_frame]
            ad[:,:,counter] =  0.2125 * frame[:,:,0] + 0.7154 * frame[:,:,1] + 0.0721 * frame[:,:,2]
            counter += 1
        return ad

    def _smoothRow(self, row, seconds_to_change = 60*30, non_transition_bins = 2, std = 100):

        ad = np.load(self._row_fn(row))
        original_shape = ad.shape

        ad[ad == 0] = 1 # 0 used for bad data to save space and use uint8 for storing data (np.nan must be a float)

        # Calculate means
        lrm = scipy.ndimage.filters.uniform_filter(ad, size = (1,self.window), mode = 'reflect', origin = -1*int(self.window/2)).astype('uint8')
        rrm = np.roll(lrm, int(self.window), axis = 1).astype('uint8')
        rrm[:,0:self.window] = lrm[:,0:1]

        # Identify data that falls outside of mean
        ad[(((ad > lrm + 7.5) & (ad > rrm + 7.5)) | ((ad < lrm - 7.5) & (ad < rrm - 7.5)))] = 0
        del lrm, rrm

        # Interpolation missing data for HMM
        ad = ad.ravel(order = 'C') #np.interp requires flattend data
        nans, x = ad==0, lambda z: z.nonzero()[0]
        ad[nans]= np.interp(x(nans), x(~nans), ad[~nans])
        del nans, x

        # Reshape array to save it
        ad = np.reshape(ad, newshape = original_shape, order = 'C').astype('uint8')
        np.save(self._row_fn(row).replace('.npy', '.smoothed.npy'), ad)
        
        return True

    def _hmmRow(self, row, seconds_to_change = 60*30, non_transition_bins = 2, std = 100, hmm_window = 60):

        data = np.load(self._row_fn(row).replace('.npy', '.smoothed.npy'))
        zs = np.zeros(shape = data.shape, dtype = 'uint8')
        for i, column in enumerate(data):

            means = scipy.ndimage.filters.uniform_filter(column, size = hmm_window, mode = 'reflect').astype('uint8')
            freq, bins = np.histogram(means, bins = range(0,257,2))
            states = bins[0:-1][freq > hmm_window]
            comp = len(states)
            if comp == 0:
                print('For row ' + str(row) + ' and column ' + str(i) + ', states = ' + str(states))
                states = [125]
            model = hmm.GaussianHMM(n_components=comp, covariance_type="spherical")
            model.startprob_ = np.array(comp*[1/comp])
            change = 1/(seconds_to_change)
            trans = np.zeros(shape = (len(states),len(states)))
            for k,state in enumerate(states):
                s_trans = np.zeros(shape = states.shape)
                n_trans_states = np.count_nonzero((states > state + non_transition_bins) | (states < state - non_transition_bins))
                if n_trans_states != 0:
                    s_trans[(states > state + non_transition_bins) | (states < state - non_transition_bins)] = change/n_trans_states
                    s_trans[states == state] = 1 - change
                else:
                    s_trans[states == state] = 1
                trans[k] = s_trans
                   
            model.transmat_ = np.array(trans)
            model.means_ = states.reshape(-1,1)
            model.covars_ = np.array(comp*[std])
            
            z = [model.means_[x][0] for x in model.predict(column.reshape(-1,1))]
            zs[i,:] = np.array(z).astype('uint8')
        np.save(self._row_fn(row).replace('.npy', '.hmm.npy'), zs)

        return True

    def _row_fn(self, row):
        return self.tempDirectory + str(row) + '.npy'

    def _print(self, outtext):
        print(str(getpass.getuser()) + ' analyzed ' + self.baseName + ' at ' + str(datetime.datetime.now()) + ' on ' + socket.gethostname() + ': ' + outtext, file = self.anLF)
        print(outtext, file = sys.stderr)
        self.anLF.close() # Close and reopen file to flush it
        subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'VideoAnalysisLog.txt', self.cloudVideoDirectory])
        self.anLF = open(self.localVideoDirectory + 'VideoAnalysisLog.txt', 'a')

    def _fixData(self, cloudMLDirectory):

        converter = {'r':'d', 'f':'t', 'o':'', 'm':'m', 'c':'', 'b':'', 'p':''}
        print('Fixing projectID: ' + self.projectID + ', Video: ' + self.baseName, file = sys.stderr)
        # This command fixes some issues with the MC6_5 cluster summary files. Zack already annotated ~2000 clips so we did not want to rerun
        if self.projectID == 'MC6_5':
            self.loadClusterSummary()

            # Fix -depth and timestamp
            self.clusterData['X_depth'] = self.clusterData.apply(lambda row: (self.transM[0][0]*row.X + self.transM[0][1]*row.Y + self.transM[0][2])/(self.transM[2][0]*row.X + self.transM[2][1]*row.Y + self.transM[2][2]), axis=1)
            self.clusterData['Y_depth'] = self.clusterData.apply(lambda row: (self.transM[1][0]*row.X + self.transM[1][1]*row.Y + self.transM[1][2])/(self.transM[2][0]*row.X + self.transM[2][1]*row.Y + self.transM[2][2]), axis=1)
            self.clusterData['TimeStamp'] = self.clusterData.apply(lambda row: (self.startTime + datetime.timedelta(seconds = int(row.t))), axis=1)

            # Add info on Zack annotation
            for row in self.clusterData.itertuples():
                LID, manualLabel, mLabeler = row.LID, row.ManualLabel, row.MLabeler
                if mLabeler == 'Zack' or (manualLabel is not np.nan and manualLabel in 'abcdefghijklmnopqrstuvwxyz'):
                    self.clusterData.loc[self.clusterData.LID == LID,'ManualAnnotation'] = 'Yes'

                if manualLabel is not np.nan and manualLabel != '' and mLabeler != 'Zack':
                    self.clusterData.loc[self.clusterData.LID == LID,'ManualLabel'] = converter[manualLabel]
                    self.clusterData.loc[self.clusterData.LID == LID,'MLabeler'] = 'Zack'
                    self.clusterData.loc[self.clusterData.LID == LID,'MLabelTime'] = datetime.datetime(month = 3, day = 5, year = 2019, hour = 12)

                # Copy videos to __Machine Learinng

            self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
            subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)


            if self.baseName == '0010_vid':
                self.createClusterClips()
                for row in self.clusterData.itertuples():
                    LID, N, t, x, y, manualAnnotation = row.LID, row.N, row.t, row.X, row.Y, row.ManualAnnotation
                    clip = str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4'
                    if manualAnnotation == 'Yes':
                        print(['rclone', 'copy', self.cloudAllClipsDirectory + clip, cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName ])
                        subprocess.call(['rclone', 'copy', self.cloudAllClipsDirectory + clip, cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName])
        


        # add code to increase number of clips for MC16_2
        if self.projectID == 'MC16_2':
            self._identifyManualClusters(Nclips = 350)
            self.createClusterClips(manualOnly = True)

