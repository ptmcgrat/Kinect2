import numpy as np
import matplotlib.pyplot as plt

class HMMdata:
    def __init__(self, width, height, frames, frameblock = 25, max_transitions = 100):
        self.data = np.empty(shape = (width * height * 50, 3), dtype = int)
        self.t = None
        self.l_diff = None
        self.abs_stop = None
        self.frameblock = frameblock #If HMM created from smaller data set, how many frames were skipped
        self.data_shape = (height, width)
        self.frames = frames
        self.current_count = 0

    def add_data(self, in_directory, out_directory, prefix = '', suffix = '.hmm.npy'):
        for i in range(self.data_shape[0]):
            data = np.load(in_directory + '/' + prefix + str(i) + suffix)
            self._add_row(data, i)
        self.data = np.delete(self.data, range(self.current_count, self.data.shape[0]), axis = 0)
        np.save(out_directory + '/HMMData.npy', self.data)

    def read_data(self, directory):
        self.data = np.load(directory + '/HMMData.npy')
              
    def _add_row(self, data, row):
        for i, column in enumerate(data):
            self._add_pixel(data[i], (row, i))
        
    def _add_pixel(self, data, position):
        cpos = 0
        out = []
        data = data
        self.split_data = np.split(data, 1 + np.where(np.diff(data) != 0)[0])
        for d in self.split_data:
            self.data[self.current_count] = (cpos, cpos + len(d) - 1, d[0])
            cpos = cpos + len(d)
            self.current_count += 1

    def ret_image(self, t):
        t = int(t/self.frameblock)
        if t > self.frames or t < 0:
            raise IndexError('Requested frame ' + str(t*frameblock) + ' doesnt exist')
        if self.t == t:
            return self.cached_frame
        else:
            self.cached_frame = self.data[(self.data[:,0] <= t) & (self.data[:,1] >= t)][:,2].reshape(self.data_shape).astype('uint8')
            self.t = t
            return self.cached_frame
            
    def ret_difference(self, start, stop, threshold = 0):
        start = int(start/self.frameblock)
        stop = int(stop/self.frameblock)
        #if (start,stop) == self.l_diff:
        #    return self.loc_changes
        indices = np.where((self.data[:,0] <= start) & (self.data[:,1] >= start))[0]
        start_data = self.data[indices]
        changes = np.zeros(shape = start_data.shape[0])
        count = 0
        while True:
            count += 1
            # Update indices for those that need it
            indices[start_data[:,1] < stop] += 1
            new_data = self.data[indices]
            diffs = np.abs(new_data[:,2] - start_data[:,2])
            if np.max(diffs) == 0:
                break
            changes[diffs > threshold] += 1
            start_data = new_data

        self.loc_changes = changes.reshape(self.data_shape)
        self.l_diff = (start,stop)
        return self.loc_changes

    def abs_difference(self, stop, threshold = 0):
        start = 0
        stop = int(stop/self.frameblock)
        #if stop == self.abs_stop:
        #    return self.abs_changes
        start_data = self.data[(self.data[:,0] <= start) & (self.data[:,1] >= start)]
        indices = np.where((self.data[:,0] <= start) & (self.data[:,1] >= start))[0]
        changes = np.zeros(shape = start_data.shape[0])
        count = 0
        while True:
            count += 1
            # Update indices for those that need it
            indices[start_data[:,1] < stop] += 1
            new_data = self.data[indices]
            diffs = np.abs(new_data[:,2] - start_data[:,2])
            if np.max(diffs) == 0:
                break
            changes[diffs > threshold] += 1
            start_data = new_data

        self.abs_changes = changes.reshape(self.data_shape)
        self.abs_stop = stop
        return self.abs_changes
