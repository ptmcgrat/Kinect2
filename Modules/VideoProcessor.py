import os, sys, psutil, subprocess, pims, datetime, shutil, cv2, math, getpass, socket, random
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pandas as pd
from hmmlearn import hmm
from Modules.HMM_data import HMMdata
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
    def __init__(self, videofile, localMasterDirectory, cloudMasterDirectory):

        # Store arguments
        self.videofile = videofile
        self.localMasterDirectory = localMasterDirectory if localMasterDirectory[-1] == '/' else localMasterDirectory + '/'
        self.cloudMasterDirectory = cloudMasterDirectory if cloudMasterDirectory[-1] == '/' else cloudMasterDirectory + '/'

        self.baseName = self.videofile.split('/')[-1].split('.')[0]
        self.localVideoDirectory = self.localMasterDirectory + 'VideoAnalysis/' + self.baseName + '/'
        self.cloudVideoDirectory = self.cloudMasterDirectory + 'VideoAnalysis/' + self.baseName + '/'
        
        self.localClusterDirectory = self.localVideoDirectory + 'ClusterData/'
        self.cloudClusterDirectory = self.cloudVideoDirectory + 'ClusterData/'

        self.tempDirectory = self.localVideoDirectory + 'Temp/'

        self.localClusterClipDirectory = self.localClusterDirectory + 'Clips/'
        self.cloudClusterClipDirectory = self.cloudClusterDirectory + 'Clips/'

        self.localLabelDirectory = self.localClusterDirectory + 'MLClips/'
        self.cloudLabelDirectory = self.cloudClusterDirectory + 'MLClips/'

        self.localAllClipsDirectory = self.localClusterDirectory + 'AllClips/'
        self.cloudAllClipsDirectory = self.cloudClusterDirectory + 'AllClips/'

        os.makedirs(self.localClusterClipDirectory) if not os.path.exists(self.localClusterClipDirectory) else None
        os.makedirs(self.localAllClipsDirectory) if not os.path.exists(self.localAllClipsDirectory) else None

        # Set paramaters
        self.cores = psutil.cpu_count() # Number of cores that should be used to analyze the video

        # Create file names
        self.hmmFile = self.baseName + '.hmm.npy'
        self.clusterFile = 'LabeledClusters.csv'
        self.labeledCoordsFile = 'LabeledCoords.npy'

        print('VideoProcessor: Analyzing ' + self.videofile, file = sys.stderr)

        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')

        self.anLF = open(self.localVideoDirectory + 'VideoAnalysisLog.txt', 'a')

    def __del__(self):
        subprocess.call(['rclone', 'copy', self.localVideoDirectory + 'VideoAnalysisLog.txt', self.cloudVideoDirectory])
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

    def loadClusterFile(self):
        try:
            self.clusterData
        except AttributeError:
            if not os.path.isfile(self.localClusterDirectory + self.clusterFile):
                subprocess.call(['rclone', 'copy', self.cloudClusterDirectory + self.clusterFile, self.localClusterDirectory], stderr = self.fnull)
                
            if not os.path.isfile(self.localClusterDirectory + self.clusterFile):
                self.createClusterHMM()
                return
            else:
                self.clusterData = pd.read_csv(self.localClusterDirectory + self.clusterFile, sep = '\t', header = 0).set_index('LID')
                
        try:
            self.labeledCoords
        except AttributeError:
            if not os.path.isfile(self.localClusterDirectory + self.labeledCoordsFile):
                subprocess.call(['rclone', 'copy', self.cloudClusterDirectory + self.labeledCoordsFile, self.localClusterDirectory], stderr = self.fnull)
                
            if not os.path.isfile(self.localClusterDirectory + self.labeledCoordsFile):
                self.createClusterHMM()
                return
            else:
                self.labeledCoords = np.load(self.localClusterDirectory + self.labeledCoordsFile)

                        
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
        self._print('calculateHMM: Converting video into HMM data')
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
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ' seconds, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]), file = sys.stderr)
            results = pool.map(self._hmmRow, rows)
        print('TotalTime: Took ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes to calculate HMMs for ' + str(self.height) + ' rows', file = sys.stderr)
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

    def createClusterHMM(self, minMagnitude = 0, treeR = 22, leafNum = 190, neighborR = 22, timeScale = 10, eps = 18, minPts = 90, delta = 1.0):
        self.loadVideo()
        self.loadHMM()
        self._print('Clustering HMM transitions using DBScan')
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

        uniqueLabels = set(self.labeledCoords[:,3])
        uniqueLabels.remove(-1)
        print(str(self.labeledCoords[self.labeledCoords[:,3] != -1].shape[0]) + ' HMM transitions assigned to ' + str(len(uniqueLabels)) + ' clusters', file = sys.stderr)

        df = pd.DataFrame(self.labeledCoords, columns=['T','X','Y','LID'])
        clusterData = df.groupby('LID').apply(lambda x: pd.Series({
            'N': x['T'].count(),
            't': int(x['T'].mean()),
            'X': int(x['X'].mean()),
            'Y': int(x['Y'].mean()),
            't_span': int(x['T'].max() - x['T'].min()),
            'X_span': int(x['X'].max() - x['X'].min()),
            'Y_span': int(x['Y'].max() - x['Y'].min()),
            'ManualLabel': '',
            'MLLabel': ''
        })
        )

        clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = '\t')
        subprocess.call(['rclone', 'sync', self.localClusterDirectory, self.cloudClusterDirectory], stderr = self.fnull)
        self.loadClusterFile()
        
    def createClusterClips(self, Nclips = 200, delta_xy = 100, delta_t = 60, smallLimit = 500):
        self.loadVideo()
        self.loadHMM()
        self.loadClusterFile()
        self._print('Creating clip videos for each cluster')

        try:
            shutil.rmtree(self.localClusterClipDirectory)
        except FileNotFoundError:
            pass
        os.makedirs(self.localClusterClipDirectory, exist_ok=True)
        try:
            shutil.rmtree(self.localLabelDirectory)
        except FileNotFoundError:
            pass
        
        os.makedirs(self.localLabelDirectory, exist_ok=True)
     
        cap = cv2.VideoCapture(self.localMasterDirectory + self.videofile)

        framerate = cap.get(cv2.CAP_PROP_FPS)

        randomizedDT = self.clusterData.sample(n = self.clusterData.shape[0])
        # Make clips for manual labeling
        manualClips = 0
        smallClips = 0
        for row in randomizedDT.itertuples():
            if manualClips > 200:
                break
            
            LID, N, t, x, y = row.Index, row.N, row.t, row.X, row.Y
            if x - delta_xy < 0 or x + delta_xy >= self.height or y - delta_xy < 0 or y + delta_xy >= self.width or LID == -1 or framerate*t - delta_t <0 or framerate*t+delta_t >= self.frames:
                continue

            if manualClips < Nclips and (N > smallLimit or smallClips < Nclips/20):
                #print(self.localClusterClipDirectory + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y) + '.mp4', file = sys.stderr)
                outManual = cv2.VideoWriter(self.localClusterClipDirectory + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), framerate, (4*delta_xy, 2*delta_xy))
                manualFlag = True
                manualClips +=1
                if N < smallLimit:
                    smallClips += 1
            else:
                manualFlag = False
                continue
                

            outAll = cv2.VideoWriter(self.localLabelDirectory + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), framerate, (2*delta_xy, 2*delta_xy))
            outAllHMM = cv2.VideoWriter(self.localLabelDirectory + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y) + '_HMM.mp4', cv2.VideoWriter_fourcc(*"mp4v"), framerate, (2*delta_xy, 2*delta_xy))


            cap.set(cv2.CAP_PROP_POS_FRAMES, int(framerate*(t) - delta_t))
            HMMChanges = self.obj.ret_difference(framerate*(t) - delta_t, framerate*(t) + delta_t)
            clusteredPoints = self.labeledCoords[self.labeledCoords[:,3] == LID][:,1:3]

            for i in range(delta_t*2):

                ret, frame = cap.read()
                frame2 = frame.copy()
                frame[HMMChanges != 0] = [300,125,125]
                for coord in clusteredPoints: # This can probably be improved to speed up clip generation (get rid of the python loop)
                    frame[coord[0], coord[1]] = [125,125,300]
                if ret:
                    if manualFlag:
                        outManual.write(np.concatenate((frame2[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy], frame[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy]), axis = 1))
                    outAll.write(frame2[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy])
                    outAllHMM.write(frame[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy])


            if outManual:
                outManual.release()
            outAll.release()
            outAllHMM.release()

        for row in self.clusterData.itertuples():
            LID, N, t, x, y = row.Index, row.N, row.t, row.X, row.Y
            if x - delta_xy < 0 or x + delta_xy >= self.height or y - delta_xy < 0 or y + delta_xy >= self.width or LID == -1 or framerate*t - delta_t <0 or framerate*t+delta_t >= self.frames:
                continue
            outAll = cv2.VideoWriter(self.localAllClipsDirectory + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), framerate, (2*delta_xy, 2*delta_xy))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(framerate*(t) - delta_t))
            for i in range(delta_t*2):
                ret, frame = cap.read()
                outAll.write(frame[x-delta_xy:x+delta_xy, y-delta_xy:y+delta_xy])
            outAll.release()
            
        subprocess.call(['rclone', 'sync', self.localClusterClipDirectory, self.cloudClusterClipDirectory], stderr = self.fnull)
        subprocess.call(['rclone', 'sync', self.localLabelDirectory, self.cloudLabelDirectory], stderr = self.fnull)
        subprocess.call(['rclone', 'sync', self.localAllClipsDirectory, self.cloudAllClipsDirectory], stderr = self.fnull)
                
        
    def labelClusters(self):

        self._print('Labeling cluster')
        self.loadClusterFile()
        subprocess.call(['rclone', 'copy', self.cloudClusterClipDirectory, self.localClusterClipDirectory], stderr = self.fnull)

        clips = [x for x in os.listdir(self.localClusterClipDirectory) if '.mp4' in x]

        categories = ['c','p','o','b','m','f','r','q']
        
        for f in clips:
            print(f)
            clusterID = int(f.split('_')[0])
            print(self.clusterData.loc[self.clusterData.index == clusterID, 'ManualLabel'])
            print(self.clusterData.loc[self.clusterData.index == clusterID, 'ManualLabel'].values[0])

            if self.clusterData.loc[self.clusterData.index == clusterID, 'ManualLabel'].values[0] in categories:
                continue
            cap = cv2.VideoCapture(self.localClusterClipDirectory + f)
            while(True):

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cv2.imshow("Type 'c' for scoop; 'p' for spit; 'o' for other; 'b' for build multiple clusters; 'm' for feed multiple; 'f' for feed spit; 'r' for spit run; 'q' to quit",cv2.resize(frame,(0,0),fx=4, fy=4))
                info = cv2.waitKey(25)
            
                if info in [ord(x) for x in categories]:
                    for i in range(1,10):
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                    break

            if info == ord('q'):
                break
            
            self.clusterData.loc[self.clusterData.index == clusterID, 'ManualLabel'] = chr(info)

        self.clusterData.to_csv(self.localClusterDirectory + self.clusterFile, sep = '\t')
        subprocess.call(['rclone', 'copy', self.localClusterDirectory + self.clusterFile, self.cloudClusterDirectory], stderr = self.fnull)
                       
    def createFramesToAnnotate(self, n = 300):
        rerun = False
        for i in range(n):
            if not os.path.isfile(self.annotationDirectory + 'AnnotateImage' + str(i).zfill(4) + '.jpg'):
                rerun = True
                break
        if not rerun:
            self._print('AnnotationFrames already created... skipping')
            return
        self.downloadVideo()
        cap = pims.Video(self.videofile)
        counter = 0
        for i in [int(x) for x in np.linspace(1.25*3600*self.frame_rate, self.frames - 1.25*3600*self.frame_rate, n)]:
            frame = cap[i]
            t_image = Image.fromarray(frame)
            t_image.save(self.annotationDirectory + 'AnnotateImage' + str(counter).zfill(4) + '.jpg')
            counter += 1
        
    def summarize_data(self):
        try:
            self.obj
        except AttributeError:
            self.obj = HMMdata(self.width, self.height, self.frames)
            self.obj.read_data(self.outdir)

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
        self.anLF = open(self.localVideoDirectory + 'VideoAnalysisLog.txt', 'a')



        
