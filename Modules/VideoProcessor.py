import pims, math, psutil, shutil, os, datetime, subprocess, sys, pickle, cv2
import numpy as np
import pandas as pd
import scipy.ndimage
from hmmlearn import hmm
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from Modules.HMM_data import HMMdata
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import NearestNeighbors
#from pathos.multiprocessing import ProcessingPool as ThreadPool
np.warnings.filterwarnings('ignore')


class VideoProcessor:
    # This class takes in an mp4 videofile and an output directory and performs the following analysis on it:
    # 1. Performs HMM analysis on all pixel files
    # 3. Clusters HMM data to identify potential spits and scoops
    # 4. Uses DeepLabCut to annotate fish

    #Parameters - blocksize, 
    def __init__(self, videofile, outDirectory, remVideoDirectory, rewrite = False):

        # Store arguments
        self.videofile = videofile
        self.baseName = self.videofile.split('/')[-1].split('.')[0]
        self.outDirectory = outDirectory if outDirectory[-1] == '/' else outDirectory + '/'
        self.remVideoDirectory = remVideoDirectory if remVideoDirectory[-1] == '/' else remVideoDirectory + '/'
        
        self.rewrite = rewrite
        
        # Set paramaters
        self.cores = psutil.cpu_count() # Number of cores that should be used to analyze the video

        # Set directories and make sure they exist
        os.makedirs(self.outDirectory) if not os.path.exists(self.outDirectory) else None

        self.hmmFile = self.outDirectory + self.baseName + '.hmm.npy'
        
        self.clusterDirectory = self.outDirectory + 'ClusterData/'
        os.makedirs(self.clusterDirectory) if not os.path.exists(self.clusterDirectory) else None
        self.annotationDirectory = self.outDirectory + 'AnnotationData/'
        os.makedirs(self.annotationDirectory) if not os.path.exists(self.annotationDirectory) else None
        self.exampleDirectory = self.outDirectory + 'ExampleData/'
        os.makedirs(self.exampleDirectory) if not os.path.exists(self.exampleDirectory) else None

        self.tempDirectory = self.outDirectory + 'Temp/'     
        shutil.rmtree(self.tempDirectory) if os.path.exists(self.tempDirectory) else None
        os.makedirs(self.tempDirectory)
      
        self.window = 120
        self.hmm_time = 60*60
        print('VideoProcessor: Analyzing ' + self.videofile, file = sys.stderr)

        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')

        

    def downloadVideo(self):
        videoName = self.videofile.split('/')[-1]
        localDirectory = self.videofile.replace(videoName,'')
        if not os.path.isfile(self.videofile):
            
            self._print(self.videofile + ' not present in local path. Trying to find it remotely')
            subprocess.call(['rclone', 'copy', self.remVideoDirectory + videoName, localDirectory], stderr = self.fnull)                
            if not os.path.isfile(self.videofile):
                self._print(self.videofile + ' not present in remote path. Trying to find h264 file and convert it to mp4')
                if not os.path.isfile(self.videofile.replace('.mp4', '.h264')):
                    subprocess.call(['rclone', 'copy', self.remVideoDirectory + videoName.replace('.mp4', '.h264'), localDirectory], stderr = self.fnull)
                if not os.path.isfile(self.videofile.replace('.mp4', '.h264')):
                    self._print('Unable to find ' + self.remVideoDirectory + videoName.replace('.mp4', '.h264'))
                    raise Exception
            
                subprocess.call(['ffmpeg', '-i', self.videofile.replace('.mp4', '.h264'), '-c:v', 'copy', self.videofile])
                
                if os.stat(self.videofile).st_size >= os.stat(self.videofile.replace('.mp4', '.h264')).st_size:
                    try:
                        vid = pims.Video(self.videofile)
                        vid.close()
                        os.remove(self.videofile.replace('.mp4', '.h264'))
                    except Exception as e:
                        self._print(e)
                        self._print('Unable to convert ' + self.videofile)
                        raise Exception
                    subprocess.call(['rclone', 'copy', self.videofile, self.remVideoDirectory], stderr = self.fnull)
                self._print(self.videofile + ' converted and uploaded to ' + self.remVideoDirectory)

        #Grab info on video
        cap = pims.Video(self.videofile)
        self.height = int(cap.frame_shape[0])
        self.width = int(cap.frame_shape[1])
        self.frame_rate = int(cap.frame_rate)
        try:
            self.frames = min(int(cap.get_metadata()['duration']*cap.frame_rate), 12*60*60*self.frame_rate)
        except AttributeError:
            self.frames = min(int(cap.duration*cap.frame_rate), 12*60*60*self.frame_rate)
        cap.close()
  
        
    def calculateHMM(self, blocksize = 5*60, delete = True):
        """
        This functon decompresses video into smaller chunks of data formated in the numpy array format.
        Each numpy array contains one row of data for the entire video.
        This function then smoothes the raw data
        Finally, an HMM is fit to the data and an HMMobject is created
        """
        print(self.hmmFile)
        print(os.path.exists(self.hmmFile))
        if os.path.exists(self.hmmFile) and not self.rewrite:
            print('Hmmfile already exists. Will not recalculate it unless rewrite flag is True')
            return

        self.downloadVideo()
        
        self.blocksize = blocksize
        total_blocks = math.ceil(self.frames/(blocksize*self.frame_rate)) #Number of blocks that need to be analyzed for the full video

        # Step 1: Convert mp4 to npy files for each row
        pool = ThreadPool(self.cores) #Create pool of threads for parallel analysis of data
        start = datetime.datetime.now()
        print('calculateHMM: Converting video into HMM data', file = sys.stderr)
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
        if delete:
            print('Converting HMMs to internal data structure and deleting temporary data', file = sys.stderr)
        else:
            print('Converting HMMs to internal data structure and keeping temporary data', file = sys.stderr)

        print('StartTime: ' + str(start), file = sys.stderr)
        
        self.obj = HMMdata(self.width, self.height, self.frames, self.frame_rate)
        self.obj.add_data(self.tempDirectory, self.hmmFile)
        # Copy example data to directory containing videofile
        subprocess.call(['cp', self._row_fn(int(self.height/2)), self._row_fn(int(self.height/2)).replace('.npy', '.smoothed.npy'), self._row_fn(int(self.height/2)).replace('.npy', '.hmm.npy'), self.exampleDirectory])

        if delete:
            shutil.rmtree(self.tempDirectory)
        print('Took ' + str((datetime.datetime.now() - start).seconds/60) + ' convert HMMs', file = sys.stderr)

    def clusterHMM(self, minMagnitude = 0, treeR = 22, leafNum = 190, neighborR = 22, timeScale = 10, eps = 18, minPts = 90, delta = 1.0):
        
        if os.path.exists(self.clusterDirectory + 'LabeledClusters.csv') and not self.rewrite:
            print('Cluster label file already exists. Will not recalculate')
            return

        if os.path.isfile(self.clusterDirectory + 'LabeledCoords.npy'):
            print('Starting from preexisting LabeledCoords.npy file', file = sys.stderr)
            labeledCoords = np.load(self.clusterDirectory + 'LabeledCoords.npy')
        else:
            if os.path.isfile(self.clusterDirectory + 'RawCoords.npy'):
                print('Starting from preexisting RawCoords.npy file', file = sys.stderr)
                coords = np.load(self.clusterDirectory + 'RawCoords.npy')
            else:
                print('Starting from HMM file', file = sys.stderr)
                try:
                    self.obj
                except AttributeError:
                    self.obj = HMMdata(filename = self.hmmFile)

                print('Creating raw coordinate file', file = sys.stderr)
                coords = self.obj.retDBScanMatrix(minMagnitude)
                np.save(self.clusterDirectory + 'RawCoords.npy', coords)
        

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
            labeledCoords = np.concatenate((sortData, labels), axis = 1).astype('int64')
            np.save(self.clusterDirectory + 'LabeledCoords.npy', labeledCoords)

        uniqueLabels = set(labeledCoords[:,3])
        uniqueLabels.remove(-1)
        print(str(labeledCoords[labeledCoords[:,3] != -1].shape[0]) + ' HMM transitions assigned to ' + str(len(uniqueLabels)) + ' clusters', file = sys.stderr)

        df = pd.DataFrame(labeledCoords, columns=['T','X','Y','LID'])
        clusterFile = df.groupby('LID').apply(lambda x: pd.Series({
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

        clusterFile.to_csv(self.clusterDirectory + 'LabeledClusters.csv', sep = '\t')

    def createClusterClipsToLabel(self, N = 100, delta_t = 100, delta_xy = 100):
        if not os.path.exists(self.clusterDirectory + 'LabeledClusters.csv'):
            #Need to run cluster analysis first
            raise Exception('Cluster file not present. Run cluster analysis first')
        
        dt = pd.read_csv(self.clusterDirectory + 'LabeledClusters.csv', sep = '\t')

        shutil.rmtree(self.clusterDirectory + 'ClusterClipsToLabel/' )
        os.makedirs(self.clusterDirectory + 'ClusterClipsToLabel/', exist_ok=True)

        self.downloadVideo()
        cap = cv2.VideoCapture(self.videofile)

        framerate = cap.get(cv2.CAP_PROP_FPS)
        for n in range(N):
            print(n)
            row = dt.sample(1)
            
            LID, N, t, x, y = row['LID'].values[0], row['N'].values[0], row['t'].values[0], row['X'].values[0], row['Y'].values[0]
            
            out = cv2.VideoWriter(self.clusterDirectory + 'ClusterClipsToLabel/' + str(LID) + '_' + str(N) + '_' + str(x) + '_' + str(y) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), framerate, (2*delta_xy, 2*delta_xy))

            print(framerate*(t) - delta_t)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(framerate*(t) - delta_t))
            for i in range(delta_t*2):
                
                ret, frame = cap.read()
                if ret:
                    out.write(frame[max(0,x-delta_xy):min(self.width,x+delta_xy), max(0,y-delta_xy):min(self.heigh,y+delta_xy)])

            out.release()
                
        
    def labelClusters(self):

        clips = [x for x in os.listdir(self.clusterDirectory + 'ClusterClipsToLabel/') if '.mp4' in x]
        dt = pd.read_csv(self.clusterDirectory + 'LabeledClusters.csv', sep = '\t')
        
        for f in clips:
            print(f)
            cap = cv2.VideoCapture(self.clusterDirectory + 'ClusterClipsToLabel/' + f)
            while(True):

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cv2.imshow("Type 'c' for scoop; 'p' for spit; 'o' for other",frame)
                info = cv2.waitKey(25)
            
                if info in [ord('c'),ord('p'),ord('o')]:
                    for i in range(1,10):
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                    break

            clusterID = int(f.split('_')[0])
            row = dt.loc[dt['LID'] == clusterID]
            row['ManualLabel'][0] = chr(info)

        dt.to_csv(self.clusterDirectory + 'LabeledClusters.csv', sep = '\t')
                
           
                
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
    
    def _readBlock(self, block):
        min_t = block*self.blocksize
        max_t = min((block+1)*self.blocksize, int(self.frames/self.frame_rate))
        ad = np.empty(shape = (self.height, self.width, max_t - min_t), dtype = 'uint8')
        cap = pims.Video(self.videofile)
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
        #       now = datetime.datetime.now()
        #        print(str(now) + ': ' + outtext, file = self.anLF)
        print(outtext, file = sys.stderr)

    def _labelCluster(cap, t, x, y, surr = 200, t_del = 50):
        frame_counter = 0
        cap.set(2, t-t_del)
        
        while(True):
            if frame_counter == 2*t_del:
                cap.set(2, t-50)
                frame_counter = 0

            ret, frame = cap.read()
            cv2.imshow("Type 'c' for scoop; 'p' for scoop; 'o' for other",frame[x-surr:x+surr, y-surr:y+surr])
            info = cv2.waitKey(1)
            
            if info in [ord('c'),ord('p'),ord('o')]:
                for i in range(1,10):
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                return chr(info)
                    
            frame_counter += 1
        

        
