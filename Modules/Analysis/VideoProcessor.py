import os, sys, psutil, subprocess, pims, datetime, shutil, cv2, math, getpass, socket, random, pdb
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pandas as pd
from hmmlearn import hmm
from Modules.Analysis.HMM_data import HMMdata
from collections import defaultdict, OrderedDict
from random import shuffle
from joblib import Parallel, delayed


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

def createClip(row, videofile, outputDirectory, frame_rate, delta_xy, delta_t):
    cap = cv2.VideoCapture(videofile)
    LID, N, t, x, y, ml = row.LID, row.N, row.t, row.X, row.Y, row.ManualAnnotation
    
    outAll = cv2.VideoWriter(outputDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (2*delta_xy, 2*delta_xy))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_rate*(t) - delta_t))
    for i in range(delta_t*2):
        ret, frame = cap.read()
        outAll.write(frame[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy])
    outAll.release()
    # return mean and std
    return True

def createClip_ffmpeg(row, videofile, outputDirectory, frame_rate, delta_xy, delta_t):
    #ffmpeg -i in.mp4 -filter:v "crop=80:60:200:100" -c:a copy out.mp4
    LID, N, t, x, y = row.LID, row.N, row.t, row.X, row.Y

    command = OrderedDict()
    command['-ss'] = str(t)
    command['-i'] = videofile
    command['-frames:v'] = str(int(2*delta_t))
    command['-filter:v'] = 'crop=' + str(2*delta_xy) + ':' + str(2*delta_xy) + ':' + str(y-delta_xy) + ':' + str(x-delta_xy)
    
    outCommand = ['ffmpeg']
    [outCommand.extend([str(a),str(b)]) for a,b in zip(command.keys(), command.values())]
    outCommand.append(outputDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4')
    print(outCommand)
    subprocess.call(outCommand, stderr = open(os.devnull, 'w'))
    #command = ['ffmpeg', '-i', self.localMasterDirectory + self.videofile, '-filter:v', 'crop=' + str(2*delta_xy) + ':' + str(2*delta_xy) + ':' + str(y-delta_xy) + ':' + str(x-delta_xy) + '', '-ss', str(t - int(delta_t/self.frame_rate)), '-frames:v', str(2*delta_t), self.localAllClipsDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4']

def createClip_scikit(row, videofile, outputDirectory, frame_rate, delta_xy, delta_t):
    LID, t, x, y = row.LID, row.t, row.X, row.Y
    rdr1 = skvideo.io.vreader(fname, inputdict={'--start_number': '0', '-vframes': '500'})

class VideoProcessor:
    # This class takes in an mp4 videofile and an output directory and performs the following analysis on it:
    # 1. Performs HMM analysis on all pixel files
    # 3. Clusters HMM data to identify potential spits and scoops
    # 4. Uses DeepLabCut to annotate fish

    #Parameters - blocksize, 
    def __init__(self, projectID, videoObj, localMasterDirectory, cloudMasterDirectory, transM, depthObj):

        # Main location for machine learning clips
        
        # Store arguments
        self.projectID = projectID
        self.videofile = videoObj.mp4_file
        self.h264_file = videoObj.h264_file
        self.movieDir = videoObj.movieDir
        self.height = videoObj.height
        self.width = videoObj.width
        self.frame_rate = videoObj.framerate
        self.startTime = videoObj.time
        self.endTime = videoObj.end_time

        self.localMasterDirectory = localMasterDirectory if localMasterDirectory[-1] == '/' else localMasterDirectory + '/'
        self.cloudMasterDirectory = cloudMasterDirectory if cloudMasterDirectory[-1] == '/' else cloudMasterDirectory + '/'
        self.transM = transM
        self.depthObj = depthObj

        self.baseName = self.videofile.split('/')[-1].split('.')[0]
        self.localVideoDirectory = self.localMasterDirectory + 'VideoAnalysis/' + self.baseName + '/'
        self.cloudVideoDirectory = self.cloudMasterDirectory + 'VideoAnalysis/' + self.baseName + '/'
        
        self.localClusterDirectory = self.localVideoDirectory + 'ClusterData/'
        self.cloudClusterDirectory = self.cloudVideoDirectory + 'ClusterData/'

        self.localCountDirectory = self.localVideoDirectory + 'Counts/'
        self.cloudCountDirectory = self.cloudVideoDirectory + 'CountsData/'

        self.tempDirectory = self.localVideoDirectory + 'Temp/'

        self.localManualLabelClipsDirectory = self.localClusterDirectory + 'ManualLabelClips/'
        self.cloudManualLabelClipsDirectory = self.cloudClusterDirectory + 'ManualLabelClips.tar'

        self.localAllClipsDirectory = self.localClusterDirectory + 'AllClips/'
        self.cloudAllClipsDirectory = self.cloudClusterDirectory + 'AllClips.tar'

        # Set paramaters
        self.cores = psutil.cpu_count() # Number of cores that should be used to analyze the video

        # Create file names
        self.hmmFile = self.baseName + '.hmm.npy'
        self.clusterFile = 'LabeledClusters.csv'
        self.labeledCoordsFile = 'LabeledCoords.npy'

        #print('VideoProcessor: Analyzing ' + self.videofile, file = sys.stderr)

        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')

        os.makedirs(self.localVideoDirectory) if not os.path.exists(self.localVideoDirectory) else None
        self.anLF = open(self.localVideoDirectory + 'VideoAnalysisLog.txt', 'a')
        print('AnalysisStart: User: ' + str(getpass.getuser()) + ',,VideoID: ' + self.baseName + ',,StartTime: ' + str(datetime.datetime.now()) + ',,ComputerID: ' + socket.gethostname(), file = self.anLF)
        self.anLF.close()

    def __del__(self):
        pass
        #subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'VideoAnalysisLog.txt', self.cloudVideoDirectory])
        #shutil.rmtree(self.localVideoDirectory)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False        
    
    def loadVideo(self, tol = 0.001):
        if os.path.isfile(self.localMasterDirectory + self.videofile):
            self._validateVideo()
            return
            #print(self.videofile + ' present in local path.', file = sys.stderr)
        
        # Try to download it from the cloud
        self._print(self.videofile + ' not present in local path. Trying to find it remotely...', log = False)
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.videofile, self.localMasterDirectory + self.videofile.split(self.videofile.split('/')[-1])[0]], stderr = self.fnull)
        if os.path.isfile(self.localMasterDirectory + self.videofile):
            self._validateVideo()
            self._print('Done', log = False)
            return

        # Try to find h264 file and convert it to mp4
        self._print('VideoConversion: ' + self.videofile + ' will be created from h264 file')
        self._print('Downloading h264 file from cloud...', log = False)
        subprocess.call(['rclone', 'copy', self.cloudMasterDirectory + self.h264_file, self.localMasterDirectory + self.movieDir], stderr = self.fnull)                
        self._print('Done', log = False)

        assert os.path.isfile(self.localMasterDirectory + self.h264_file)

        # Convert it using ffmpeg
        command = ['ffmpeg', '-r', str(self.frame_rate), '-i', self.localMasterDirectory + self.h264_file, '-c:v', 'copy', '-r', str(self.frame_rate), self.localMasterDirectory + self.videofile]
        self._print('VideoConversion: ' + ' '.join(command))
        subprocess.call(command, stderr = self.fnull)
        
        # Ensure the conversion went ok.     
        assert os.stat(self.localMasterDirectory + self.videofile).st_size >= os.stat(self.localMasterDirectory + self.h264_file).st_size

        self._validateVideo(log = True)
        
        # Make pdf showing brighness over time
        current_t = self.startTime
        total_minutes = 0
        times = []
        brightness = []
        cap = cv2.VideoCapture(self.localMasterDirectory + self.videofile)

        while current_t < self.endTime:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.frame_rate*(total_minutes*60)))
            ret, frame = cap.read()

            times.append(current_t)
            brightness.append(frame.mean(axis = (0,1)))

            total_minutes += 1
            current_t += datetime.timedelta(minutes = 1)

        cap.release()

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots()
        ax.plot(times, [x[0] for x in brightness], color = 'b')
        ax.plot(times, [x[1] for x in brightness], color = 'g')
        ax.plot(times, [x[2] for x in brightness], color = 'r')

        ax.set_title(self.baseName + ' brightness over time')

        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Brightness')

        fig.savefig(self.localVideoDirectory + 'Brightness.pdf', dpi=300)

        self._print('Uploading ' + self.videofile + ' to cloud...', log = False)
        subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'Brightness.pdf', self.cloudVideoDirectory], stderr = self.fnull)
        subprocess.Popen(['rclone', 'copy', self.localMasterDirectory + self.videofile, self.cloudMasterDirectory + self.movieDir], stderr = self.fnull)

        self._print('Done', log = False)

    def _validateVideo(self, tol = 0.001, log = False):
        assert os.path.isfile(self.localMasterDirectory + self.videofile)
        
        cap = cv2.VideoCapture(self.localMasterDirectory + self.videofile)
        new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        new_framerate = cap.get(cv2.CAP_PROP_FPS)
        new_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        predicted_frames = int((self.endTime - self.startTime).total_seconds()*self.frame_rate)

        if log:
            self._print('VideoValidation: Size: ' + str((new_height,new_width)) + ',,fps: ' + str(new_framerate) + ',,Frames: ' + str(new_frames) + ',,PredictedFrames: ' + str(predicted_frames), log = log)

        assert new_height == self.height
        assert new_width == self.width
        assert abs(new_framerate - self.frame_rate) < tol*self.frame_rate
        assert abs(predicted_frames - new_frames) < tol*predicted_frames

        self.frames = new_frames

        cap.release()

    def loadHMM(self):
        #print('Loading HMM', file = sys.stderr)
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
        #print('Loading Clusters', file = sys.stderr)
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
        #print('Loading ClusterSummary', file = sys.stderr)
        try:
            self.clusterData
        except AttributeError:
            #if not os.path.isfile(self.localClusterDirectory + self.clusterFile):
            print(['rclone', 'copy', self.cloudClusterDirectory + self.clusterFile, self.localClusterDirectory])
            subprocess.call(['rclone', 'copy', self.cloudClusterDirectory + self.clusterFile, self.localClusterDirectory], stderr = self.fnull)
                
            if not os.path.isfile(self.localClusterDirectory + self.clusterFile):
                self.createClusterSummary()
                return
            else:
                self.clusterData = pd.read_csv(self.localClusterDirectory + self.clusterFile, sep = ',', header = 0, index_col = 0)
                                  
    def createHMM(self, blocksize = 5*60, window = 120):
        """
        This functon decompresses video into smaller chunks of data formated in the numpy array format.
        Each numpy array contains one row of data for the entire video.
        This function then smoothes the raw data
        Finally, an HMM is fit to the data and an HMMobject is created
        """
        
        #Download video
        self.loadVideo()

        shutil.rmtree(self.tempDirectory) if os.path.exists(self.tempDirectory) else None
        os.makedirs(self.tempDirectory)
        
        self.blocksize = blocksize # Number of seconds that are decompressed at a time by a single thread
        self.window = window # Size of rolling average for mean for smoothing analysis

        maxTime = self.startTime.replace(hour = 18, minute = 0, second = 0, microsecond = 0) # Lights dim at 6pm. 

        self.HMMframes = min(self.frames, int((maxTime - self.startTime).total_seconds()*self.frame_rate))
        #self.hmm_time = hmm_time
        
        total_blocks = math.ceil(self.HMMframes/(blocksize*self.frame_rate)) #Number of blocks that need to be analyzed for the full video

        # Step 1: Convert mp4 to npy files for each row
        pool = ThreadPool(self.cores) #Create pool of threads for parallel analysis of data
        start = datetime.datetime.now()
        self._print('HMMCreation: Outfile: ' + self.hmmFile + ',,Blocksize(seconds): ' + str(blocksize) + ',,Window: ' + str(window))
        self._print('HMMCreation: FramesUsed: ' + str(self.HMMframes) + ',,TotalBlocks: ' + str(total_blocks) + ',,TotalThreads: ' + str(self.cores))
        #print('TotalThreads: ' + str(self.cores), file = sys.stderr)
        #print('Video processed: ' + str(self.blocksize/60) + ' min per block, ' + str(self.blocksize/60*self.cores) + ' min per cycle', file = sys.stderr)
        self._print('HMMCreation: Converting mp4 data to npy arrays at 1 fps')
        self._print('StartTime: ' + str(start), log = False)
        
        for i in range(0, math.ceil(total_blocks/self.cores)):
            blocks = list(range(i*self.cores, min(i*self.cores + self.cores, total_blocks)))
            self._print('Minutes since start: ' + str((datetime.datetime.now() - start).seconds/60) + ', Processing blocks: ' + str(blocks[0]) + ' to ' +  str(blocks[-1]), log = False)
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
        self._print('TotalTime: ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes', log = False)

        # Step 2: Smooth data to remove outliers
        pool = ThreadPool(self.cores)
        start = datetime.datetime.now()
        self._print('HMMCreation: Smoothing data to filter out outliers')
        #print('StartTime: ' + str(start), file = sys.stderr)
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            self._print('Minutes since start: ' + str((datetime.datetime.now() - start).seconds/60) + ', Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]), log = False)
            results = pool.map(self._smoothRow, rows)
        self._print('TotalTime: Took ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes to smooth ' + str(self.height) + ' rows', log = False)
        pool.close() 
        pool.join()

        # Step 3: Calculate HMM values for each row
        pool = ThreadPool(self.cores)
        start = datetime.datetime.now()
        self._print('HMMCreation: Calculating HMMs for all data')
        #print('StartTime: ' + str(start), file = sys.stderr)
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            print('Hours since start: ' + str((datetime.datetime.now() - start).seconds/3600) + ' hours, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]), file = sys.stderr)
            results = pool.map(self._hmmRow, rows)
        print('TotalTime: Took ' + str((datetime.datetime.now() - start).seconds/3600) + ' hours to calculate HMMs for ' + str(self.height) + ' rows', file = sys.stderr)
        pool.close() 
        pool.join()
        
        # Step 4: Create HMM object and delete temporary data if necessary
        start = datetime.datetime.now()
        self._print('Converting HMMs to internal data structure and keeping temporary data', log = False)

        #self._print('StartTime: ' + str(start), file = sys.stderr)
        
        self.obj = HMMdata(self.width, self.height, self.HMMframes, self.frame_rate)
        self.obj.add_data(self.tempDirectory, self.localVideoDirectory + self.hmmFile)
        # Copy example data to directory containing videofile
        subprocess.call(['cp', self._row_fn(int(self.height/2)), self._row_fn(int(self.height/2)).replace('.npy', '.smoothed.npy'), self._row_fn(int(self.height/2)).replace('.npy', '.hmm.npy'), self.localVideoDirectory])

        shutil.rmtree(self.tempDirectory)
      
        subprocess.call(['rclone', 'copy', self.localVideoDirectory, self.cloudVideoDirectory], stderr = self.fnull)

        self._print('HMMCreation: Complete')

    def createClusters(self, minMagnitude = 0, treeR = 22, leafNum = 190, neighborR = 22, timeScale = 10, eps = 18, minPts = 90, delta = 1.0):
        #self.loadVideo()
        self.loadHMM()
        
        self._print('ClusterCreation: File: ' + self.labeledCoordsFile + ',,MinMagnitude: ' + str(minMagnitude) + ',,treeR: ' + str(treeR) + ',,LeafNum: ' + str(leafNum))
        self._print('ClusterCreation: NeighborR: ' + str(neighborR) + ',,timescale: ' + str(timeScale) + ',,eps: ' + str(eps) + ',,minPts: ' + str(minPts))

        coords = self.obj.retDBScanMatrix(minMagnitude)
        np.save(self.localClusterDirectory + 'RawCoords.npy', coords)
        #subprocess.call(['rclone', 'copy', self.localClusterDirectory + 'RawCoordsFile.npy', self.cloudClusterDirectory], stderr = self.fnull)
        self._print('RawCoordinates calculated', log = False)     

        sortData = coords[coords[:,0].argsort()][:,0:3] #sort data by time for batch processing, throwing out 4th column (magnitude)
        numBatches = int(sortData[-1,0]/delta/3600) + 1 #delta is number of hours to batch together. Can be fraction.

        sortData[:,0] = sortData[:,0]*timeScale #scale time so that time distances between transitions are comparable to spatial differences
        labels = np.zeros(shape = (sortData.shape[0],1), dtype = sortData.dtype)

        #Calculate clusters in batches to avoid RAM overuse
        curr_label = 0 #Labels for each batch start from zero - need to offset these
            
        self._print('ClusterCreation: TotalBatches: ' + str(numBatches))
        for i in range(numBatches):
            self._print('Batch: # ' + str(i), log = False)
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
        self._print('Sycncing with cloud', log = False)
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.labeledCoordsFile, self.cloudClusterDirectory], stderr = self.fnull)
        self._print('ClusterCreation: Completed')

    def createClusterSummary(self, Nclips = 400, startHour = 8, stopHour = 18, delta_xy = 100, delta_t = 60, smallLimit = 500):
        #self.loadVideo()
        #self.loadHMM()
        self.loadClusters()
        self._print('ClusterSummaryCreation: File: ' + self.clusterFile + ',,Nclips: ' + str(Nclips))
        uniqueLabels = set(self.labeledCoords[:,3])
        uniqueLabels.remove(-1)
        self._print('ClusterSummaryCreation: TotalHMMTransitions: ' + str(self.labeledCoords.shape[0]) + ',,AssignedHMMTransitions: ' + str(self.labeledCoords[self.labeledCoords[:,3] != -1].shape[0]) + ',,NumClusters: ' + str(len(uniqueLabels)))
        self._print('ClusterSummaryCreation: deltaXY: ' + str(delta_xy) + ',,deltaT: ' + str(delta_t) + ',,startHour: ' + str(startHour) + ',,stopHour: ' + str(stopHour) + ',,smallLimit: ' + str(smallLimit))

        minTime = self.startTime.replace(hour = startHour, minute = 0, second = 0, microsecond = 0)
        maxTime = self.startTime.replace(hour = stopHour, minute = 0, second = 0, microsecond = 0)

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
            'ClipCreated': 'No',
            'DepthChange': np.nan,
        })
        )
        self._print('Calculating X and Y positions with respect to Kinect coordinate system', log = False)
        clusterData['Y_depth'] = clusterData.apply(lambda row: (self.transM[0][0]*row.Y + self.transM[0][1]*row.X + self.transM[0][2])/(self.transM[2][0]*row.Y + self.transM[2][1]*row.X + self.transM[2][2]), axis=1)
        clusterData['X_depth'] = clusterData.apply(lambda row: (self.transM[1][0]*row.Y + self.transM[1][1]*row.X + self.transM[1][2])/(self.transM[2][0]*row.Y + self.transM[2][1]*row.X + self.transM[2][2]), axis=1)
        clusterData['TimeStamp'] = clusterData.apply(lambda row: (self.startTime + datetime.timedelta(seconds = int(row.t))), axis=1)
        
        clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        clusterData = pd.read_csv(self.localClusterDirectory + self.clusterFile, sep = ',', header = 0)

        # Identify clusters to make clips for
        self._print('Identifying clusters to make clips for', log = False)
        smallClips, clipsCreated = 0,0 # keep track of clips with small number of pixel changes
        for row in clusterData.sample(n = clusterData.shape[0]).itertuples(): # Randomly go through the dataframe
            LID, N, t, x, y, time, xDepth, yDepth = row.LID, row.N, row.t, row.X, row.Y, datetime.datetime.strptime(row.TimeStamp, '%Y-%m-%d %H:%M:%S.%f'), int(row.X_depth), int(row.Y_depth)
            try:
                currentDepth = self.depthObj._returnHeightChange(self.depthObj.lp.frames[0].time, time)[xDepth,yDepth]
            except IndexError: # x and y values are outside of depth field of view
                currentDepth = np.nan
            clusterData.loc[clusterData.LID == LID,'DepthChange'] = currentDepth
            # Check spatial compatability:
            if x - delta_xy < 0 or x + delta_xy >= self.height or y - delta_xy < 0 or y + delta_xy >= self.width:
                continue
            # Check temporal compatability (part a):
            elif self.frame_rate*t - delta_t < 0 or LID == -1:
                continue
                #print('Cannot create clip for: ' + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y), file = sys.stderr)
                clusterData.loc[clusterData.LID == LID,'ClipCreated'] = 'No'
            # Check temporal compatability (part b):
            elif time < minTime or time > maxTime:
                continue
            else:
                clusterData.loc[clusterData.LID == LID,'ClipCreated'] = 'Yes'
                if N < smallLimit:
                    if smallClips > Nclips/20:
                        continue
                    smallClips += 1
                if clipsCreated < Nclips:
                    clusterData.loc[clusterData.LID == LID,'ManualAnnotation'] = 'Yes'
                    clipsCreated += 1

        clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        self.clusterData = clusterData
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)

    def createClusterClips(self, delta_xy = 100, delta_t = 60):
        self.loadVideo()
        self.loadHMM()
        self.loadClusters()
        self.loadClusterSummary()
        self._print('ClipCreation: Starting')

        shutil.rmtree(self.localManualLabelClipsDirectory) if os.path.exists(self.localManualLabelClipsDirectory) else None
        os.makedirs(self.localManualLabelClipsDirectory)

        shutil.rmtree(self.localAllClipsDirectory) if os.path.exists(self.localAllClipsDirectory) else None
        os.makedirs(self.localAllClipsDirectory)

        #self._createMean()
        #cap = pims.Video(self.localMasterDirectory + self.videofile)

        # Clip creation is super slow so we do it in parallel
        processedVideos = Parallel(n_jobs=self.cores)(delayed(createClip)(row, self.localMasterDirectory + self.videofile, self.localAllClipsDirectory, self.frame_rate, delta_xy, delta_t) for row in self.clusterData[self.clusterData.ClipCreated == 'Yes'].itertuples())
        self._print('ClipCreation: ClipsCreated: ' + str(len(processedVideos)))

        # Calculate mean and standard deviations of clip videos

        #subprocess.call(['rclone', 'copy', self.localVideoDirectory + self.meansFile, self.cloudVideoDirectory], stderr = self.fnull)
        self._print('ClipCreation: AllClipsCreated')
        # Now create manual label clips, which require extra data
        cap = cv2.VideoCapture(self.localMasterDirectory + self.videofile)

        mlClips = 0
        for row in self.clusterData[self.clusterData.ManualAnnotation == 'Yes'].itertuples():
            LID, N, t, x, y = row.LID, row.N, row.t, row.X, row.Y
            
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
                outAllHMM.write(np.concatenate((frame2[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy], frame[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy]), axis = 1))

            
            outAllHMM.release()
            mlClips += 1

            subprocess.call(['cp', self.localAllClipsDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4', self.localManualLabelClipsDirectory])
            assert(os.path.exists(self.localManualLabelClipsDirectory + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4'))
        cap.release()

        self._print('ClipCreation: ManualLabelClipsCreated: ' + str(mlClips) + ',,Syncying...')
        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)
        
        subprocess.call(['tar', '-cvf', self.localManualLabelClipsDirectory[:-1] + '.tar', '-C', self.localClusterDirectory, self.localManualLabelClipsDirectory.split('/')[-2]], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.localManualLabelClipsDirectory[:-1] + '.tar', self.cloudClusterDirectory], stderr = self.fnull)
        subprocess.call(['tar', '-cvf', self.localAllClipsDirectory[:-1] + '.tar', '-C', self.localClusterDirectory, self.localAllClipsDirectory.split('/')[-2]], stderr = self.fnull)
        subprocess.call(['rclone', 'copy', self.localAllClipsDirectory[:-1] + '.tar', self.cloudClusterDirectory], stderr = self.fnull)
        self._print('ClipCreation: Finished')

    def loadClusterClips(self, allClips = True, mlClips = False):
        if allClips:
            subprocess.call(['rclone', 'copy', self.cloudAllClipsDirectory, self.localClusterDirectory], stderr = self.fnull)
            subprocess.call(['tar', '-C', self.localClusterDirectory, '-xvf', self.localAllClipsDirectory[:-1] + '.tar'], stderr = self.fnull)
        if mlClips:
            subprocess.call(['rclone', 'copy', self.cloudManualLabelClipsDirectory, self.localClusterDirectory], stderr = self.fnull)
            subprocess.call(['tar', '-C', self.localClusterDirectory, '-xvf', self.localManualLabelClipsDirectory[:-1] + '.tar'], stderr = self.fnull)

    def labelClusters(self, rewrite, mainDT, cloudMLDirectory, number):

        self._print('ManualLabelCreation: ClustersRequested: ' + str(number))
        self.loadClusterSummary()
        self.loadClusterClips(allClips = False, mlClips = True)

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
        categories = ['c','f','p','t','b','m','s','x','o','d','q', 'k']

        print("Type 'c': build scoop; 'f': feed scoop; 'p': build spit; 't': feed spit; 'b': build multiple; 'm': feed multiple; 'd': drop sand; s': spawn; 'o': fish other; 'x': nofish other; 'q': quit; 'k': skip")
        
        newClips = []
        annotatedClips = 0
        #pdb.set_trace()

        shuffle(clips)
        for f in clips:
            clusterID = int(f.split('_')[0])

            # If already labeled and rewrite = False, then skip
            if not rewrite:
                label = self.clusterData.loc[self.clusterData.LID == clusterID].ManualLabel.values[0]
                if label == label:
                    # is not np.nan
                    if label in ''.join(categories):
                        print('Skipping ' + f + ': Label=' + label, file = sys.stderr)
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
            
            if info == ord('k'):
                continue #skip

            self.clusterData.loc[self.clusterData.LID == clusterID, 'ManualLabel'] = chr(info)
            self.clusterData.loc[self.clusterData.LID == clusterID, 'MLabeler'] = socket.gethostname()
            self.clusterData.loc[self.clusterData.LID == clusterID, 'MLabelTime'] = str(datetime.datetime.now())
            self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')

            newClips.append(f.replace('_ManualLabel',''))
            annotatedClips += 1

            print(['rclone', 'copy', self.localManualLabelClipsDirectory + f.replace('_ManualLabel',''), cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName])
            subprocess.Popen(['rclone', 'copy', self.localManualLabelClipsDirectory + f.replace('_ManualLabel',''), cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName], stderr = self.fnull)
            if number is not None and annotatedClips > number:
                break

        self._print('Syncing labeled data with cloud', log=False)

        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)

        subprocess.call(['rclone', 'copy', cloudMLDirectory + mainDT, self.localClusterDirectory], stderr = self.fnull)
        tempData = pd.read_csv(self.localClusterDirectory + mainDT, sep = ',', header = 0, index_col = 0)
        tempData2 = pd.concat([tempData, self.clusterData[self.clusterData.ManualLabel != ''].dropna(subset=['ManualLabel'])], sort = False)
        
        tempData2.drop_duplicates(subset=['projectID', 'videoID', 'LID'], inplace=True, keep='last')

        tempData2.to_csv(self.localClusterDirectory + mainDT, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + mainDT, cloudMLDirectory], stderr = self.fnull)

        self._print('ManualLabelCreation: ClustersLabeled: ' + str(annotatedClips))

    def predictLabels(self, modelLocation, modelIDs):
        from Modules.Analysis.MachineLabel import MachineLearningMaker as MLM
        self.loadClusterSummary()
        return self.clusterData
        print('Creating model object')
        #subprocess.call(['rclone', 'copy', modelLocation + 'classInd.txt', self.localVideoDirectory], stderr = self.fnull)
        #subprocess.call(['rclone', 'copy', modelLocation + 'model.pth', self.localVideoDirectory], stderr = self.fnull)
        
        MLobj = MLM('', [''], self.localVideoDirectory, modelLocation, self.cloudAllClipsDirectory, labeledClusterFile = None, classIndFile = None)
        MLobj.prepareData()
        labels = MLobj.predictLabels(modelIDs)

        for label in labels:
            self.clusterData = pd.merge(self.clusterData, label, on = ['LID', 'N'], how = 'left')
        
        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)

        return self.clusterData

    def cleanup(self):
        shutil.rmtree(self.localVideoDirectory)
        if os.path.exists(self.localMasterDirectory + self.videofile):
            os.remove(self.localMasterDirectory + self.videofile)
        subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'VideoAnalysisLog.txt', self.cloudVideoDirectory])
         
    def countFish(self, rewrite, cloudCountDirectory, nFrames = 500):
        print(cloudCountDirectory)
        self.loadVideo()
        categories = list(range(0,10))
        frames = set()
        counts = defaultdict(int)
        total = 0
        while True:
            if total % 50 == 0:
                print(total)
                print(counts)
            if total > nFrames:
                break
            frame = random.randint(0,self.frames-1)
            if frame in frames:
                continue
            frames.add(frame)
            #pic = self._retFrame(frame, noBackground = False)
            pic, num = self._retFrame(frame, noBackground = True)
            if counts['0'] ==200 and counts[1] == 200 and num < 18000:
                continue
            elif counts['1'] == 200 and num > 8000 and num < 18000:
                continue
            pic2 = self._retFrame(frame, noBackground = False)
            cv2.imshow("How many fish are in frame " + str(frame) + "? Pixels = " + str(num) + ". 'q': quit",cv2.resize(pic,(0,0),fx=4, fy=4))
            info = cv2.waitKey(25)
            while info not in [ord(str(x)) for x in categories]:
                info = cv2.waitKey(25)
            
            for j in range(1,10):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            count = chr(info)
            if counts[count] == 200:
                continue
            counts[count] += 1
            total += 1
            outDirectory = self.localCountDirectory + str(chr(info)) + '/'
            os.makedirs(outDirectory) if not os.path.exists(outDirectory) else None
            cv2.imwrite(outDirectory + 'Frame_'+ self.projectID + '_' + self.baseName + '_' + str(frame) + '.jpg', pic2)
            if info == ord('q'):
                break

        subprocess.call(['rclone', 'copy', self.localCountDirectory, cloudCountDirectory + self.projectID + '/' + self.baseName + '/'])

    def _retFrame(self, frameNum, noBackground = True, cutoff = 15):
        try:
            self.cap
        except AttributeError:
            self.loadVideo()
            self.cap = cv2.VideoCapture(self.localMasterDirectory + self.videofile)

        if noBackground:
            self.loadHMM()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception('Cant read frame number: ' + str(frameNum))

        if noBackground:
            frame = 0.2125 * frame[:,:,0] + 0.7154 * frame[:,:,1] + 0.0721 * frame[:,:,2]
            HMMChanges = self.obj.ret_image(frameNum)
            diff = frame - HMMChanges
            diff[(diff>cutoff) | (diff < -1*cutoff)] = 125
            diff[diff!=125] = 0
            #pdb.set_trace()
            return np.concatenate((frame.astype("uint8"),diff.astype("uint8")), axis = 1), np.count_nonzero(diff)
        else:
            return frame

    def _readBlock(self, block):
        min_t = block*self.blocksize
        max_t = min((block+1)*self.blocksize, int(self.HMMframes/self.frame_rate))
        ad = np.empty(shape = (self.height, self.width, max_t - min_t), dtype = 'uint8')
        
        cap = cv2.VideoCapture(self.localMasterDirectory + self.videofile)
        for i in range(max_t - min_t):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i+min_t)*self.frame_rate))
            ret, frame = cap.read()
            if not ret:
                raise Exception('Cant read frame')
            ad[:,:,i] =  0.2125 * frame[:,:,2] + 0.7154 * frame[:,:,1] + 0.0721 * frame[:,:,0] #opencv does bgr instead of rgb
        cap.release()
        return ad 

        #cap = pims.Video(self.localMasterDirectory + self.videofile)
        #counter = 0
        #for i in range(min_t, max_t):
        #    current_frame = int(i*self.frame_rate)
        #    frame = cap[current_frame]
        #    ad[:,:,counter] =  0.2125 * frame[:,:,0] + 0.7154 * frame[:,:,1] + 0.0721 * frame[:,:,2]
        #    counter += 1
        #cap.close()
        #return ad

    def _smoothRow(self, row):

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

    def _print(self, outtext, log = True):
        if log:
            self.anLF = open(self.localVideoDirectory + 'VideoAnalysisLog.txt', 'a')
            print('  ' + outtext + ',,Time: ' + str(datetime.datetime.now()), file = self.anLF)
            self.anLF.close()
            subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'VideoAnalysisLog.txt', self.cloudVideoDirectory])

        print(outtext, file = sys.stderr)

    def _fixData(self, cloudMLDirectory):

        maxTime = self.startTime.replace(hour = 18, minute = 0, second = 0, microsecond = 0)

        self.loadClusterSummary()
        #MC16_2 and TI2_4 run at the wrong frame rate
        if self.projectID == 'MC16_2':
            self.clusterData['TimeStamp'] = self.clusterData.apply(lambda row: (self.startTime + datetime.timedelta(seconds = int(row.t*25))), axis=1)
        if self.projectID == 'TI2_4':
            if self.baseName == '0004_vid':
                return
            self.clusterData['TimeStamp'] = self.clusterData.apply(lambda row: (self.startTime + datetime.timedelta(seconds = int(row.t*25))), axis=1)

        for row in self.clusterData.itertuples():
            LID, N, t, x, y, time, manualAnnotation, xDepth, yDepth = row.LID, row.N, row.t, row.X, row.Y, datetime.datetime.strptime(row.TimeStamp, '%Y-%m-%d %H:%M:%S.%f'), row.ManualAnnotation, int(row.X_depth), int(row.Y_depth)
            try:
                currentDepth = self.depthObj._returnHeightChange(self.depthObj.lp.frames[0].time, time)[xDepth,yDepth]
            except IndexError: # x and y values are outside of depth field of view
                currentDepth = np.nan
            self.clusterData.loc[self.clusterData.LID == LID,'DepthChange'] = currentDepth

            if time > maxTime:
                self.clusterData.loc[self.clusterData.LID == LID, 'ClipCreated'] = 'No'
            if manualAnnotation == 'Yes':
                self.clusterData.loc[self.clusterData.LID == LID, 'ManualAnnotation'] = 'No'
                subprocess.call(['rclone', 'delete', cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName + '/' + str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4'])

        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)


        """
        for row in self.clusterData.itertuples():
            LID, N, t, x, y, manualAnnotation, manualLabel = row.LID, row.N, row.t, row.X, row.Y, row.ManualAnnotation, row.ManualLabel
            if manualAnnotation == 'No':
                continue
            elif manualLabel != manualLabel:
                continue
            else:
                clip = str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4'
                subprocess.call(['rclone', 'copy', self.cloudAllClipsDirectory + clip, cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName])
        """
        #self._createMean()
        #self.loadClusterSummary()
        #self.clusterData['Y_depth'] = self.clusterData.apply(lambda row: (self.transM[0][0]*row.Y + self.transM[0][1]*row.X + self.transM[0][2])/(self.transM[2][0]*row.Y + self.transM[2][1]*row.X + self.transM[2][2]), axis=1)
        #self.clusterData['X_depth'] = self.clusterData.apply(lambda row: (self.transM[1][0]*row.Y + self.transM[1][1]*row.X + self.transM[1][2])/(self.transM[2][0]*row.Y + self.transM[2][1]*row.X + self.transM[2][2]), axis=1)
        #self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
        #subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)
        #self._addHeightChange(depthObject)
        """

        converter = {'r':'d', 'f':'t', 'o':'', 'm':'m', 'c':'', 'b':'', 'p':''}
        print('Fixing projectID: ' + self.projectID + ', Video: ' + self.baseName, file = sys.stderr)
        #self._addHeightChange(depthObject)
        # This command fixes some issues with the MC6_5 cluster summary files. Zack already annotated ~2000 clips so we did not want to rerun
        if self.projectID == 'MC6_5':
            self.loadClusterSummary()

            # Fix -depth and timestamp
            #self.clusterData['X_depth'] = self.clusterData.apply(lambda row: (self.transM[0][0]*row.X + self.transM[0][1]*row.Y + self.transM[0][2])/(self.transM[2][0]*row.X + self.transM[2][1]*row.Y + self.transM[2][2]), axis=1)
            #self.clusterData['Y_depth'] = self.clusterData.apply(lambda row: (self.transM[1][0]*row.X + self.transM[1][1]*row.Y + self.transM[1][2])/(self.transM[2][0]*row.X + self.transM[2][1]*row.Y + self.transM[2][2]), axis=1)
            #self.clusterData['TimeStamp'] = self.clusterData.apply(lambda row: (self.startTime + datetime.timedelta(seconds = int(row.t))), axis=1)

            # Add info on Zack annotation
            for row in self.clusterData.itertuples():
                LID, N, t, x, y, manualAnnotation, clipCreated = row.LID, row.N, row.t, row.X, row.Y, row.ManualAnnotation, row.ClipCreated
                if clipCreated == 'No':
                    continue
                self.clusterData.loc[self.clusterData.LID == LID,'ClipCreated'] = 'Yes'
                # Change name of file:
                oldclip = str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y) + '.mp4'
                newclip = str(LID) + '_' + str(N) + '_' + str(t) + '_' + str(x) + '_' + str(y) + '.mp4'
                print(['rclone','moveto', self.cloudAllClipsDirectory + oldclip, self.cloudAllClipsDirectory + newclip])
                subprocess.call(['rclone','moveto', self.cloudAllClipsDirectory + oldclip, self.cloudAllClipsDirectory + newclip])
                if manualAnnotation == 'Yes':   
                    subprocess.call(['rclone', 'copy', self.cloudAllClipsDirectory + newclip, cloudMLDirectory + 'Clips/' + self.projectID + '/' + self.baseName])
                    
            self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = ',')
            subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)

      
                #LID, manualLabel, mLabeler = row.LID, row.ManualLabel, row.MLabeler
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
            else:
                self.createClusterClips(manualOnly = True)


        # add code to increase number of clips for MC16_2
        #if self.projectID == 'MC16_2':
        #    self._identifyManualClusters(Nclips = 350)
        #    self.createClusterClips(manualOnly = True)
        """

