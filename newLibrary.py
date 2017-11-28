import cv2, datetime, sys, os, subprocess, seaborn
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import animation
import matplotlib.pyplot as plt

#import pylibfreenect2 as FN2

class Kinect2Tracker:
    def __init__(self, project_name):

        # 1: Set parameters
        self.master_start = datetime.datetime.now()
        self.frame_counter = 1 # Keep track of how many frames are saved
        self.background_counter = 1 # Keep track of how many backgrounds are saved
        self.stdev_threshold = 20 # Maximum standard deviation to keep pixel values
        self.max_background_time = datetime.timedelta(seconds = 3600) # Maximum time before calculating new background
        np.seterr(invalid='ignore')
        self.caff = subprocess.Popen('caffeinate')
        
        # 2: Create master directory and logging files
        self.master_directory = '/Users/pmcgrath7/Dropbox (GaTech)/Applications/KinectPiProject/Kinect2Tests/Output/' + project_name + '/'
        if not os.path.exists(self.master_directory):
            os.mkdir(self.master_directory)
        self.logger_file = self.master_directory + 'Logfile.txt'
        self.lf = open(self.logger_file, 'w')
        print('MasterStart: ' + str(self.master_start), file = self.lf)
        print('MasterStart: ' + str(self.master_start), file = sys.stderr)
        
        # 3: Open and start Kinect2
        self.start_kinect()

        # 4: Identify ROI for depth data study
        self.create_ROI()

        # 5: Diagnose speed
        self.diagnose_speed()

        # 6: Grab initial background
        self.create_background()

    def __del__(self):
        print('ObjectDestroyed: ' + str(datetime.datetime.now()), file = self.lf)
        print('ObjectDestroyed: ' + str(datetime.datetime.now()), file = sys.stderr)
        self.lf.close()
        self.caff.kill()
        
    def start_kinect(self):
        # a: Identify pipeline to use: 1) OpenGL, 2) OpenCL, 3) CPU
        try:
            self.pipeline = FN2.OpenGLPacketPipeline()
        except:
            try:
                self.pipeline = FN2.OpenCLPacketPipeline()
            except:
                self.pipeline = FN2.CpuPacketPipeline()
        print('PacketPipeline: ' + type(self.pipeline).__name__, file = self.lf)
        print('PacketPipeline: ' + type(self.pipeline).__name__, file = sys.stderr)

        # b: Create and set logger
        self.logger = FN2.createConsoleLogger(FN2.LoggerLevel.NONE)
        FN2.setGlobalLogger(self.logger)

        # c: Identify device and create listener
        self.fn = FN2.Freenect2()
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            print("DeviceError: No device connected!", file = self.lf)
            print("DeviceError: No device connected!", file = sys.stderr)
            sys.exit(1)

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial, pipeline=self.pipeline)

        self.listener = FN2.SyncMultiFrameListener(
            FN2.FrameType.Color | FN2.FrameType.Ir | FN2.FrameType.Depth)

        # d: Register listeners
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)

        # e: Start device and create registration
        self.device.start()
        self.registration = FN2.Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())

    def create_ROI(self):
        # a: Grab color and depth frames and register them
        undistorted = FN2.Frame(512, 424, 4)
        registered = FN2.Frame(512, 424, 4)
        frames = self.listener.waitForNewFrame()
        color = frames["color"]
        depth = frames["depth"]
        self.registration.apply(color, depth, undistorted, registered)
        reg_image =  registered.asarray(np.uint8)
        
        # b: Select ROI using open CV
        self.r = cv2.selectROI('Image', reg_image)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        self.listener.release(frames)

        # c: Save file with background rectangle
        cv2.rectangle(reg_image, (self.r[0], self.r[1]), (self.r[0] + self.r[2], self.r[1]+self.r[3]) , (0, 255, 0), 2)
        cv2.imwrite(self.master_directory+'BoundingBox.jpg', reg_image)

        print('ROI: Bounding box created, Image: BoundingBox.jpg, Shape: ' + str(self.r), file = self.lf)
        print('ROI: Bounding box created, Image: BoundingBox.jpg, Shape: ' + str(self.r), file = sys.stderr)
        
    def diagnose_speed(self, time = 10):
        delta = datetime.timedelta(seconds = time)
        start_t = datetime.datetime.now()
        counter = 0
        while True:
            frames = self.listener.waitForNewFrame()
            depth = frames["depth"]
            data = depth.asarray()[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
            counter += 1
            self.listener.release(frames)
            if datetime.datetime.now() - start_t > delta:
                break
        print('DiagnoseSpeed: Captured ' + str(counter) + ' frames in ' + str(time) + ' seconds.', file = self.lf)
        print('DiagnoseSpeed: Captured ' + str(counter) + ' frames in ' + str(time) + ' seconds.', file = sys.stderr)

    def create_background(self, num_frames = 3, save = True):
        print('Capturing Background', file = sys.stderr)
        self.background_time = datetime.datetime.now()
        background_data = np.empty(shape = (5, self.r[3], self.r[2]))
        background_data[:] = np.NAN
        for i in range(0,5):
            background_data[i] = self.capture_frame(time = 20, delta = 0.2, save = False)
        self.background = np.nanmedian(background_data, axis = 0)
        std = np.nanstd(background_data, axis = 0)
        self.background[std > self.stdev_threshold] = np.nan
        
        if save:
            print('BackgroundCaptured: Background_' + str(self.background_counter).zfill(4) + '.npy, ' + str(self.background_time) + ', Med: '+ '%.2f' % np.nanmean(self.background) + ', Std: ' + '%.2f' % np.nanmean(std) + ', GP: ' + str(np.count_nonzero(~np.isnan(self.background)))  + ' of ' +  str(self.background.shape[0]*self.background.shape[1]), file = self.lf)
            print('BackgroundCaptured: Background_' + str(self.background_counter).zfill(4) + '.npy, ' + str(self.background_time) + ', Med: '+ '%.2f' % np.nanmean(self.background) + ', Std: ' + '%.2f' % np.nanmean(std) + ', GP: ' + str(np.count_nonzero(~np.isnan(self.background)))  + ' of ' +  str(self.background.shape[0]*self.background.shape[1]), file = sys.stderr)

            np.save(self.master_directory + 'Background_' + str(self.background_counter).zfill(4) + '.npy', self.background)
            self.background_counter += 1

    def capture_frame(self, time = 60, delta = 0.5, save = True, background = False):
        delta = datetime.timedelta(seconds = delta)
        time = datetime.timedelta(seconds = time)

        if datetime.datetime.now() - self.background_time > self.max_background_time:
            self.create_background()

        #Create array to hold data
        all_data = np.empty(shape = (int(time/delta), self.r[3], self.r[2]))
        all_data[:] = np.NAN
        
        #self.device.start()
        counter = -1
        #Collect data
        # For each received frame...
        start_t = datetime.datetime.now()
        last_t = start_t
        frame = FN2.Frame(512, 424, 4)

        #print('FrameCapture: Capturing ~' + str(int(time/delta)) + ' frames: ', end = '', file = sys.stderr)
        while True:
            frames = self.listener.waitForNewFrame()
            depth = frames["depth"]
            if (datetime.datetime.now() - last_t) >= delta:
                if counter == -1:
                    counter += 1
                    self.listener.release(frames)
                    continue
                else:
                    #print(str(counter + 1) + ' ', end = '', file = sys.stderr)
                    #sys.stderr.flush()
                    last_t = datetime.datetime.now()
                    data = depth.asarray()[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
                    data[data == 0] = np.nan
                    all_data[counter] = data
                    counter += 1
                    if (last_t - start_t) > time:
                        self.listener.release(frames)
                        break
            self.listener.release(frames)

        #print('',file = sys.stderr)
        med = np.nanmedian(all_data, axis = 0)
        std = np.nanstd(all_data, axis = 0)
        med[std > self.stdev_threshold] = np.nan
        
        counts = np.count_nonzero(~np.isnan(all_data), axis = 0)
        med[counts < 2] = np.nan
#        print('\n\tAvg med: ' + '%.2f' % np.nanmean(med), end = '')
#        print('. Avg std: ' + '%.2f' % np.nanmean(std), end = '')
#        print('. Good pixels: ' + str(np.count_nonzero(~np.isnan(med)))  + ' of ' +  str(med.shape[0]*med.shape[1]), end = '')
#        print('. FC: ' + str(counter))
        if save:
            print('FrameCaptured: Frame_' + str(self.frame_counter).zfill(4) + '.npy, ' + str(start_t) + ', Med: '+ '%.2f' % np.nanmean(med) + ', Std: ' + '%.2f' % np.nanmean(std) + ', Min: ' + '%.2f' % np.nanmin(med) + ', Max: ' + '%.2f' % np.nanmax(med) + ', GP: ' + str(np.count_nonzero(~np.isnan(med)))  + ' of ' +  str(med.shape[0]*med.shape[1]), file = self.lf)
            print('FrameCaptured: Frame_' + str(self.frame_counter).zfill(4) + '.npy, ' + str(start_t) + ', Med: '+ '%.2f' % np.nanmean(med) + ', Std: ' + '%.2f' % np.nanmean(std) + ', Min: ' + '%.2f' % np.nanmin(med) + ', Max: ' + '%.2f' % np.nanmax(med) + ', GP: ' + str(np.count_nonzero(~np.isnan(med)))  + ' of ' +  str(med.shape[0]*med.shape[1]), file = sys.stderr)
            np.save(self.master_directory +'Frame_' + str(self.frame_counter).zfill(4) + '.npy', med)
            self.frame_counter += 1
        return med

    def capture_frames(self, total_time = 60*60*24*1/24):
        
        delta = datetime.timedelta(seconds = total_time)

        while True:
            if datetime.datetime.now() - self.master_start > delta:
                break
            self.capture_frame()

class Kinect2Analyzer:
    def __init__(self, project_name):

        # 1: Set parameters and parse Logger file
        self.master_directory = '/Users/pmcgrath7/Dropbox (GaTech)/Applications/KinectPiProject/Kinect2Tests/Output/' + project_name + '/'
        self.logger_file = self.master_directory + 'Logfile.txt'
        self.lf = open(self.logger_file, 'r')
        
    def parse_log(self):

        self.npy_files = []
        self.meds = []
        self.mins = []
        self.maxes = []
        self.stds = []
        self.pixels = []
        
        for line in self.lf:
            info_type = line.split(':')[0]
            if info_type == 'FrameCaptured':
                npy_file = line.split(': ')[1].split(',')[0]
                fmed = float(line.split('Med: ')[1].split(',')[0])
                #fmin = float(line.split('Min: ')[1].split(',')[0])
                #fmax = float(line.split('Max: ')[1].split(',')[0])
                fstd = float(line.split('Std: ')[1].split(',')[0])
                fpix = (int(line.split('GP: ')[1].split(' of')[0]), int(line.rstrip().split(' of ')[1]))

                self.npy_files.append(self.master_directory + npy_file)
                self.meds.append(fmed)
                #self.mins.append(fmin)
                #self.maxes.append(fmax)
                self.stds.append(fstd)
                self.pixels.append(fpix)

            if info_type == 'ROI':
                self.background_image = cv2.imread(self.master_directory + line.rstrip().split('Image: ')[1].split(',')[0])
                self.crop_shape = (int(line.rstrip().split('Shape: (')[1].split(',')[0]), int(line.rstrip().split('Shape: (')[1].split(',')[1].split(')')[0])) 

        self.all_data = np.empty(shape = (len(self.npy_files),) + self.crop_shape)

        for i, npy_file in enumerate(self.npy_files):
            data = np.load(npy_file)
            self.all_data[i] = data

    def create_heatmap_video(self):

        self.d_min = np.nanmin(self.all_data)
        self.d_max = np.nanmax(self.all_data)
        self.d_min2 = np.nanmin(self.all_data - self.all_data[0])
        self.d_max2 = np.nanmax(self.all_data - self.all_data[0])
        
        fig = plt.figure()
        im = seaborn.heatmap(kt_obj.all_data[0])
        writer = animation.writers['ffmpeg'](fps=3, metadata=dict(artist='Me'), bitrate=1800)
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init, frames = self.all_data.shape[0], repeat = False)
        anim.save('test.mp4', writer = writer)
        anim2 = animation.FuncAnimation(fig, self.animate2, init_func=self.init2, frames = self.all_data.shape[0], repeat = False)
        anim2.save('test2.mp4', writer = writer)
        
    def init(self):
        seaborn.heatmap(self.all_data[0], vmin = self.d_min, vmax = self.d_max)

    def animate(self, i):
        plt.clf()
        seaborn.heatmap(self.all_data[i], vmin = self.d_min, vmax = self.d_max)
        
    def init2(self):
        seaborn.heatmap(self.all_data[0] - self.all_data[0], vmin = self.d_min2, vmax = self.d_max2)

    def animate2(self, i):
        plt.clf()
        seaborn.heatmap(self.all_data[i] - self.all_data[0], vmin = self.d_min2, vmax = self.d_max2)

            
kt_obj = Kinect2Analyzer('Test')
kt_obj.parse_log()
kt_obj.create_heatmap_video()

    
#kt_obj = Kinect2Tracker('Test2')
#kt_obj.capture_frames()

