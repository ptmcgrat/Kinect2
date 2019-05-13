import numpy as np
import matplotlib.pyplot as plt
import sys

class HMMdata:
    def __init__(self, width = None, height = None, frames = None, frameblock = 25, filename = None, max_transitions = 100):
        if filename is not None:
            self.read_data(filename)
        else:
            if width is None or height is None or frames is None:
                print(width)
                print(height)
                print(frames)
                raise TypeError('If filename is not provided, width, height, and frames must be provided')
            self.data_shape = (height, width)
            self.frameblock = frameblock #If HMM created from smaller data set, how many frames were skipped
            self.frames = frames
            self.height = height
            self.width = width
            self.data = np.empty(shape = (width * height * 50, 3), dtype = int)
            
        self.t = None
        self.l_diff = None
        self.abs_stop = None
        self.current_count = 0

    def add_data(self, in_directory, outfile, prefix = '', suffix = '.hmm.npy'):
        # This function creates a 1D numpy array of tuples (
        for i in range(self.data_shape[0]):
            data = np.load(in_directory + '/' + prefix + str(i) + suffix)
            self._add_row(data, i)
        self.data = np.delete(self.data, range(self.current_count, self.data.shape[0]), axis = 0)

        self.write_data(outfile)
        
    def write_data(self, outfile):
        if '.npy' not in outfile:
            outfile += '.npy'
        np.save(outfile, self.data)
        with open(outfile.replace('.npy','.txt'), 'w') as f:
            print('Width: ' + str(self.data_shape[1]), file = f)
            print('Height: ' + str(self.data_shape[0]), file = f)
            print('Frames: ' + str(self.frames), file = f)
            print('FrameBlock: ' + str(self.frameblock), file = f)
 
            
    def read_data(self, infile):
        self.data = np.load(infile)
        with open(infile.replace('.npy','.txt')) as f:
            for line in f:
                if line.rstrip() != '':
                    data, value = line.rstrip().split(': ')
                    value = int(value)
                    if data == 'Width':
                        self.width = value
                    if data == 'Height':
                        self.height = value
                    if data == 'Frames':
                        self.frames = value
                    if data == 'FrameBlock':
                        self.frameblock = value
        self.data_shape = (self.height, self.width)

    def retDBScanMatrix(self, minMagnitude = 0, densityFilter = 1):
        #minMagnitude is the size of the color change for a pixel to need to have
        #densityFilter filters out time points that have to many changes occur across the frame (1 = 1% of all pixels)

        print(str(self.data.shape[0] - self.width*self.height) + ' raw transitions', file = sys.stderr)
        
        #Identify total pixel changes in a unit of time
        time, counts = np.unique(self.data[:,0], return_counts = True)
        threshold = counts[0]*densityFilter/100
        
        row = 0
        column = -1
        allCoords = np.zeros(shape = (int(self.data.shape[0] - self.width*self.height), 4), dtype = 'uint64')
        i = 0
        for d in self.data:
            if d[0] == 0:
                column+=1
                if column == self.width:
                    column = 0
                    row += 1
            else:
                numChanges = counts[np.where(time==d[0])[0][0]]
                if numChanges < threshold:
                    allCoords[i] = np.array((d[0], row, column, abs(d[2] - prev_mag)), dtype = 'uint64')
                    i+=1
            prev_mag = d[2]
            
        allCoords = allCoords[allCoords[:,3] > minMagnitude].copy()
        print(str(allCoords.shape[0]) + ' HMM transitions passed filtering criteria', file = sys.stderr)
        return allCoords
            
    def _add_row(self, data, row):
        for i, column in enumerate(data):
            self._add_pixel(data[i], (row, i))
        
    def _add_pixel(self, data, position):
        cpos = 0
        out = []
        data = data
        self.split_data = np.split(data, 1 + np.where(np.diff(data) != 0)[0])
        for d in self.split_data:
            try:
                self.data[self.current_count] = (cpos, cpos + len(d) - 1, d[0])
            except IndexError: # numpy array is too small to hold all the data. Resize it
                self.data = np.resize(self.data, (self.data.shape[0]*5, self.data.shape[1]))
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

    def mag_density(self, frame, x0 = 20, t0 = 60):
        outdata = []
        
        diffs = abs(self.ret_image(frame).astype(int) - self.ret_image(max(frame - self.frameblock, 0)).astype(int))
        dens = self.ret_difference(max(frame - t0, 0), min(frame+t0,self.frames))
        xs, ys = np.where(diffs!=0)

        for x,y in zip(xs,ys):
            density = dens[max(0,x-x0): min(x+x0,self.width), max(0,y-x0): min(y+x0, self.height)].sum()
            outdata.append((diffs[x,y], density))

        return outdata
            
    def summarizeData(self):
        diffs = self.ret_difference(0, self.frames)
        start_frame = 0
        stop_frame = min(self.frames)
        pass
        
