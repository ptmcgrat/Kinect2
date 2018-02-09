import pims, math, psutil, shutil, os, datetime
import numpy as np
import scipy.ndimage
from hmmlearn import hmm
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from Modules.HMM_data import HMMdata

#from pathos.multiprocessing import ProcessingPool as ThreadPool
np.warnings.filterwarnings('ignore')


class VideoProcessor:
    def __init__(self, videofile, blocksize = 60*5):
        self.videofile = videofile
        self.blocksize = blocksize # in seconds

        #Grab info on video
        cap = pims.Video(self.videofile)
        self.height = int(cap.frame_shape[0])
        self.width = int(cap.frame_shape[1])
        self.frame_rate = int(cap.frame_rate)
        self.frames = min(int(cap.duration*cap.frame_rate), 12*60*60*self.frame_rate)
        cap.close()
        
        self.cores = psutil.cpu_count()
        self.total_blocks = math.ceil(self.frames/(self.blocksize*self.frame_rate))
        self.outdir = videofile.split(videofile.split('/')[-2])[0] + 'Analysis/'
        self.tempdir =  os.path.expanduser('~') + '/Temp/KinectTracker/' + videofile.split('/')[-1].split('.')[0] + '/'
        self.window = 120
        self.hmm_time = 60*60
        print('VideoProcessor: Analyzing ' + self.videofile)


    def convertVideo(self):
        """ 
        This functon decompresses video into smaller chunks of data formated in the numpy array format.
        Each numpy array contains one row of data for the entire video.
        """
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
        os.makedirs(self.tempdir)
        pool = ThreadPool(self.cores)
        print('ConvertingVideo: Creating rows of data for further analysis')
        print('TotalBlocks: ' + str(self.total_blocks))
        print('TotalThreads: ' + str(self.cores))
        print('Video processed: ' + str(self.blocksize/60) + ' min per block, ' + str(self.blocksize/60*self.cores) + ' min per cycle')
        start = datetime.datetime.now()
        print('StartTime: ' + str(start))
        
        for i in range(0, math.ceil(self.total_blocks/self.cores)):
            blocks = list(range(i*self.cores, min(i*self.cores + self.cores, self.total_blocks)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ', Processing blocks: ' + str(blocks[0]) + ' to ' +  str(blocks[-1]))
            results = pool.map(self._readBlock, blocks)
            print('Data read: ' + str((datetime.datetime.now() - start).seconds) + ' seconds')
            for row in range(self.height):
                row_file = self.row_fn(row)
                out_data = np.concatenate([results[x][row] for x in range(len(results))], axis = 1)
                if os.path.isfile(row_file):
                    out_data = np.concatenate([np.load(row_file),out_data], axis = 1)
                np.save(row_file, out_data)
            print('Data wrote: ' + str((datetime.datetime.now() - start).seconds) + ' seconds' )
        pool.close() 
        pool.join() 
        print('TotalTime: ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes')
            

    def calculateHMM(self, delete = True):
        
        start = datetime.datetime.now()
        print('ConvertingVideo: Creating rows of data for further analysis')
        print('TotalThreads: ' + str(self.cores))
        pool = ThreadPool()
        print('Smoothing ' + str(self.height) + ' total rows')
        print('StartTime: ' + str(start))
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ' seconds, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]))
            results = pool.map(self._smoothRow, rows)
        print('Took ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes to smooth ' + str(self.height) + ' rows')
        pool.close() 
        pool.join() 
        pool = ThreadPool(self.cores)

        start = datetime.datetime.now()
        print('Calculating HMMs for ' + str(self.height) + ' total rows')
        print('StartTime: ' + str(start))
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ' seconds, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]))
            results = pool.map(self._hmmRow, rows)
        print('Took ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes to calculate HMMs for ' + str(self.height) + ' rows')
        pool.close() 
        pool.join()
        start = datetime.datetime.now()
        print('Converting HMMs to internal data structure and deleting temporary data')
        start = datetime.datetime.now()
        print('StartTime: ' + str(start))

        
        self.obj = HMMdata(self.width, self.height, self.frames)
        if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)    
        self.obj.add_data(self.tempdir, self.outdir)
        #if delete:
        #    shutil.rmtree(self.tempdir)
        print('Took ' + str((datetime.datetime.now() - start).seconds/60) + ' convert HMMs')
        
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
    
    def _readBlock(self,block):
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

        ad = np.load(self.row_fn(row))
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
        np.save(self.row_fn(row).replace('.npy', '.smoothed.npy'), ad)
        
        return True

    def _hmmRow(self, row, seconds_to_change = 60*30, non_transition_bins = 2, std = 100, hmm_window = 60):

        data = np.load(self.row_fn(row).replace('.npy', '.smoothed.npy'))
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
        np.save(self.row_fn(row).replace('.npy', '.hmm.npy'), zs)

        return True

    def row_fn(self, row):
        return self.tempdir + str(row) + '.npy'

