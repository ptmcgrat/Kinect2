from freenect2 import Device, FrameType

class Kinect2Tracker():
    def __init__(self):
        self.device = Device()
        self.frame_length = 60 #Time in seconds

    def collect_
        
    def mode(self, ndarray, axis=0):
        # Modified from https://stackoverflow.com/questions/16330831/most-efficient-way-to-find-mode-in-numpy-array
        # Check inputs
        ndarray = numpy.asarray(ndarray)
        ndim = ndarray.ndim
        if ndarray.size == 1:
            return (ndarray[0], 1)
        elif ndarray.size == 0:
            raise Exception('Cannot compute mode on empty array')
        try:
            axis = range(ndarray.ndim)[axis]
        except:
            raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

        # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
        if all([ndim == 1, int(numpy.__version__.split('.')[0]) >= 1, int(numpy.__version__.split('.')[1]) >= 9]):
            modals, counts = numpy.unique(ndarray, return_counts=True)
            index = numpy.argmax(counts)
            return modals[index], counts[index]

        # Sort array
        sort = numpy.sort(ndarray, axis=axis)
        # Create array to transpose along the axis and get padding shape
        transpose = numpy.roll(numpy.arange(ndim)[::-1], axis)
        shape = list(sort.shape)
        shape[axis] = 1
        # Create a boolean array along strides of unique values
        strides = numpy.concatenate([numpy.zeros(shape=shape, dtype='bool'),
                                 numpy.diff(sort, axis=axis) == 0,
                                 numpy.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
        # Count the stride lengths
        counts = numpy.cumsum(strides)
        counts[~strides] = numpy.concatenate([[0], numpy.diff(counts[~strides])])
        counts[strides] = 0
        # Get shape of padded counts and slice to return to the original shape
        shape = numpy.array(sort.shape)
        shape[axis] += 1
        shape = shape[transpose]
        slices = [slice(None)] * ndim
        slices[axis] = slice(1, None)
        # Reshape and compute final counts
        counts = counts.reshape(shape).transpose(transpose)[slices] + 1

        # Find maximum counts and return modals/counts
        slices = [slice(None, i) for i in sort.shape]
        del slices[axis]
        index = numpy.ogrid[slices]
        index.insert(axis, numpy.argmax(counts, axis=axis))
        return sort[index], counts[index]
        
class KinectTracker():
    def __init__(self):


        
        self.background_count = 1
        self.bad_data = 2047
        self.threshold_value = 5
        self.fish_padding = 10
        self.initial_background_t = 60 # seonds
        self.delta_t = 0.4 #seconds
        self.stdev_thresh = 0.2
        self.change_thresh = 10
        self.update_time = 30
        self.current_background = self.identify_valid_masked_background()
        self.background_time=time.time()
        self.total_background = np.zeros(shape=(480,640))
        self.total_masks = np.zeros(shape=(480,640))
        self.total_updates = 0
        self.update_thresh_count = 60
        self.update_thresh_time = 60
        self.record_time = 10 * 60 * 60
        self.record_rate = 5 * 60

        
    def calculate_masked_background(self):
        #testing
        gc.enable()
        print gc.isenabled()
        cf0,_ = freenect.sync_get_depth() #Grab a frame
        self.all_data = cf0[np.newaxis,...] #Initialize 3D array
        ts0 = time.time() # Capture starting time
        ts = ts0
        count = 0
        while(ts - ts0 < self.initial_background_t): # Fix logic to convert time stamp to seconds
            cf,_ = freenect.sync_get_depth()
            ts = time.time()
            if ts > count*self.delta_t + ts0:
                self.all_data = np.vstack([self.all_data, cf[np.newaxis,...]])
                count += 1
        print(ts - ts0)
        masked_data = np.ma.masked_where(self.all_data == self.bad_data, self.all_data)
        print "datatype masked_data before:"
        print masked_data.dtype
        masked_data = masked_data.astype(np.float16)
        print masked_data.shape
        print sys.getsizeof(self.all_data)
        print "datatype masked_data after:"
        print masked_data.dtype
        collected = gc.collect()
        print "Garbage collector: collected %d objects." % (collected)
        st = np.ma.std(masked_data, axis = 0, dtype=np.float16)
        print "std datatype:"
        print st.dtype
        #end testing
        
        num_masked = np.sum(masked_data.mask == True)
        num_greater = np.sum(st > self.stdev_thresh)
        #PM#3 - Create high stdev pixels to mask (> self.stdev_thresh) -masked_data2
        masked_data2 = np.ma.masked_where(st > self.stdev_thresh, st)
        #PM#4 - Calculate masked median of both, excluding <2047 values - masked_data_median
        masked_both = np.ma.masked_where(np.logical_or(masked_data.mask, masked_data2.mask), self.all_data)
        masked_data_median = np.ma.median(masked_both, axis=0)
        return masked_data_median
        gc.disable()
    def identify_valid_masked_background(self, display = False):
        #PM#4 Calculate three examples of masked background
        count = 1
        med_sum = np.empty(shape=(480,640))[np.newaxis, ...]
        mask_sum = np.empty(shape=(480,640))[np.newaxis, ...]
        while count <= 3:
            med_mask = self.calculate_masked_background()
            med_sum = np.vstack([med_sum, med_mask.data[np.newaxis,...]])
            mask_sum = np.vstack([mask_sum, med_mask.mask[np.newaxis,...]])
            print 'Background Calculated'
            count += 1
            time.sleep(60)
        med_sum=med_sum[1:4,:,:]
        mask_sum=1-(mask_sum[1:4,:,:])
        #PM#5 Combine these into one background
        # Average 3 values where both data is not masked
        elements = np.multiply(med_sum,mask_sum)
        total_values = np.sum(elements,0)
        valid_pixels = np.sum(mask_sum,0)
        valid_pixels[valid_pixels==0] = 1
        AverageBackground = np.divide(total_values, valid_pixels)
        #PM#6 Print background to file
        np.save('Initial Background', AverageBackground)
        #PM#7 Display (bad pixels = black or white) if display = True
        if display == True:
            BadPixels = np.zeros(shape=(480,640))
            BadPixels[valid_pixels==0] = 1
            cv2.imshow('Bad Pixels', BadPixels)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return AverageBackground
        
    def average_background(self, display = False):
        ts0 = time.time() # Capture starting time
        ts = ts0
        valcount=np.zeros(shape=(480,640))
        pixcount=np.zeros(shape=(480,640))
        while(ts - ts0 < self.update_time):
            cf, t = freenect.sync_get_depth()
            cf_masked = np.ma.masked_where(cf == self.bad_data, cf)
            diff = (cf_masked - self.current_background).astype('uint8')
            # threshold diff
            ret, thresh = cv2.threshold(diff, self.change_thresh, 255, cv2.THRESH_BINARY)
            # dilate image
            strel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(thresh, strel, iterations=1)
            #identify objects and bounding boxes
            if display == True:
                copy=dilated
                (cnts, _) = cv2.findContours(copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
					(x, y, w, h) = cv2.boundingRect(c)
					cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Fish Objects', copy)
                cv2.waitKey(1)
            #mask out fish objects as well
            mask_both = np.ma.masked_where(np.logical_or(cf==self.bad_data, dilated == True), cf)
            #add good data to sum and count
            log_mask=np.zeros(shape=(480,640))
            log_mask[mask_both.mask == False] = 1
            self.total_background += np.multiply(mask_both.data,log_mask)
            valcount += np.multiply(mask_both.data,log_mask)
            self.total_masks += log_mask
            pixcount += log_mask
            self.total_updates += 1
            ts = time.time()
        #Calculate average height - note: includes outside bower unless bounding regions are somehow applied
        average_height = np.divide(valcount,pixcount)
        #Save to appropriate place
        dt = datetime.datetime.now().strftime(" %m-%d-%Y")
        name = 'Background - ' + str(self.background_count) + dt
        np.save(name, average_height)
		upCommand = '/home/pi/Dropbox-Uploader/dropbox_uploader.sh upload ' + maindir + date + '_vid.mp4' + ' /Tank41/' + date + '_vid.mp4'
		up = call([upCommand], shell=True)
        self.background_count += 1
        #Update log file
        logging.info('Generated a background at:' + dt)
        #Update background if appropriate
        if np.logical_or((self.total_updates >= self.update_thresh_count), (time.time() - self.background_time) >= self.update_thresh_time):
            new_background = np.divide(self.total_background,self.total_masks)
            self.background_time=time.time()
            self.current_background = new_background
            self.total_background = np.zeros(shape=(480,640))
            self.total_masks = np.zeros(shape=(480,640))
            self.total_updates = 0

    def long_term_recording(self):
        seconds = 0
        while seconds < self.record_time:
            s_time=time.time()
            elapsed = (s_time-camtime)/60
            logging.info('[INFO] Elapsed recording time:' + str(elapsed) + ' minutes')
            self.average_background()
            seconds += self.record_rate
            e_time = time.time() - s_time
            
            time.sleep(self.record_rate - e_time)


    def mode(self,ndarray,axis=0):
        if ndarray.size == 1:
            return (ndarray[0],1)
        elif ndarray.size == 0:
            raise Exception('Attempted to find mode on an empty array!')
        try:
            axis = [i for i in range(ndarray.ndim)][axis]
        except IndexError:
            raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ndarray.ndim))
        srt = np.sort(ndarray,axis=axis)
        dif = np.diff(srt,axis=axis)
        shape = [i for i in dif.shape]
        shape[axis] += 2
        indices = np.indices(shape)[axis]
        index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
        indices[index][dif == 0] = 0
        indices.sort(axis=axis)
        bins = np.diff(indices,axis=axis)
        location = np.argmax(bins,axis=axis)
        mesh = np.indices(bins.shape)
        index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
        index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
        counts = bins[tuple(index)].reshape(location.shape)
        index[axis] = indices[tuple(index)]
        modals = srt[tuple(index)].reshape(location.shape)
        return (modals, counts)
