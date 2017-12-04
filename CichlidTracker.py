import cv2, platform, sys, os, shutil, datetime, subprocess
import numpy as np

class CichlidTracker:
    def __init__(self, project_name, output_directory, rewrite_flag):

        # 1: Set parameters
        self.master_start = datetime.datetime.now()
        self.frame_counter = 1 # Keep track of how many frames are saved
        self.background_counter = 1 # Keep track of how many backgrounds are saved
        self.stdev_threshold = 20 # Maximum standard deviation to keep pixel values
        self.max_background_time = datetime.timedelta(seconds = 3600) # Maximum time before calculating new background
        self.kinect_bad_data = 2047
        np.seterr(invalid='ignore')

        # 2: Determine which system this code is running on
        if platform.node() == 'odroid':
            self.system = 'odroid'
        elif platform.node() == 'raspberrypi':
            self.system = 'pi'
        elif platform.system() == 'Darwin':
            self.system = 'mac'
            self.caff = subprocess.Popen('caffeinate') #Prevent mac from falling asleep
        else:
            print('Not sure what system you are running on.', file = sys.stderr)
            sys.exit()

        # 3: Determine which Kinect is attached
        self.identify_device() #Stored in self.device

        # 4: Determine if PiCamera is attached
        self.PiCamera = False
        if self.system == 'pi':
            from picamera import PiCamera
            self.camera = PiCamera()
            self.camera.resolution = (1296, 972)
            self.camera.framerate = 30

        # 5: Create master directory and logging files
        if output_directory is None:
            if self.system == 'odroid':
                self.master_directory = '/media/odroid/Kinect2/' + project_name + '/'
            elif self.system == 'pi':
                self.master_directory = '/media/pi/Kinect2/' + project_name + '/'
            elif self.system == 'mac':
                self.master_directory = '/Users/pmcgrath7/Dropbox (GaTech)/Applications/KinectPiProject/Kinect2Tests/Output/' + project_name + '/'
        else:
            self.master_directory = output_directory + project_name + '/'

        if rewrite_flag:
            if os.path.exists(self.master_directory):
                shutil.rmtree(self.master_directory)
            
        if not os.path.exists(self.master_directory):
            os.mkdir(self.master_directory)
        else:
            print('Project directory already exists. If you would like to overwrite, use the -r flag')
            sys.exit()
            
        self.logger_file = self.master_directory + 'Logfile.txt'
        self.lf = open(self.logger_file, 'w')
        self._print('MasterStart: Time=' + str(self.master_start) + ',System=' + self.system + ',Device=' + self.device + ',Camera=' + str(self.PiCamera))
        self._print('MasterStart: Uname=' + str(platform.uname()))
        
        # 6: Open and start Kinect2
        self.start_kinect()

        # 7: Identify ROI for depth data study
        self.create_ROI()

        # 8: Diagnose speed
        self.diagnose_speed()
        
        # 9: Grab initial background
        self.create_background()

    def __del__(self):
        print('ObjectDestroyed: ' + str(datetime.datetime.now()), file = self.lf)
        print('ObjectDestroyed: ' + str(datetime.datetime.now()), file = sys.stderr)
        self.lf.close()
        if self.device == 'kinect2':
            self.K2device.stop()
        if self.device == 'kinect':
            freenect.sync_stop()

        if self.device == 'mac':
            self.caff.kill()

    def _print(self, text):
        print(text, file = self.lf)
        print(text, file = sys.stderr)

    def _return_reg_color(self):
        if self.device == 'kinect':
            return freenect.sync_get_video()[0]

        elif self.device == 'kinect2':
            undistorted = FN2.Frame(512, 424, 4)
            registered = FN2.Frame(512, 424, 4)
            frames = self.listener.waitForNewFrame()
            color = frames["color"]
            depth = frames["depth"]
            self.registration.apply(color, depth, undistorted, registered, enable_filter=False)
            reg_image =  registered.asarray(np.uint8)[:,:,0:3]
            self.listener.release(frames)
            return reg_image

    def _return_depth(self):
        if self.device == 'kinect':
            data = freenect.sync_get_depth()[0].astype('Float64')
            data[data == self.kinect_bad_data] = np.nan
            return data
        
        elif self.device == 'kinect2':
            frames = self.listener.waitForNewFrame()
            output = frames['depth'].asarray()
            self.listener.release(frames)
            return output

    def _video_recording(self):
        if datetime.datetime.now().hour > 8 and datetime.datetimenow.hour < 20:
            return True
        else:
            return False
        
    def identify_device(self):
        try:
            global freenect
            import freenect
            a = freenect.init()
            if freenect.num_devices(a) == 0:
                kinect = False
            elif freenect.num_devices(a) > 1:
                print('Multiple Kinect1s attached. Unsure how to handle', file = sys.stderr)
                sys.exit()
            else:
                kinect = True
        except ImportError:
            kinect = False

        try:
            global FN2
            import pylibfreenect2 as FN2
            if FN2.Freenect2().enumerateDevices() == 1:
                kinect2 = True
            elif FN2.Freenect2().enumerateDevices() > 1:
                print('Multiple Kinect2s attached. Unsure how to handle', file = sys.stderr)
                sys.exit()
            else:
                kinect2 = False
        except ImportError:
            kinect2 = False

        if kinect and kinect2:
            print('Kinect1 and Kinect2 attached. Unsure how to handle', file = sys.stderr)
            sys.exit()
        elif not kinect and not kinect2:
            print('No device attached. Quitting...', file = sys.stderr)
            sys.exit()
        elif kinect:
            self.device = 'kinect'
        else:
            self.device = 'kinect2'

    def start_kinect(self):
        if self.device == 'kinect':
            freenect.sync_get_depth() #Grabbing a frame initializes the device
            freenect.sync_get_video()

        elif self.device == 'kinect2':
            # a: Identify pipeline to use: 1) OpenGL, 2) OpenCL, 3) CPU
            try:
                self.pipeline = FN2.OpenCLPacketPipeline()
            except:
                try:
                    self.pipeline = FN2.OpenGLPacketPipeline()
                except:
                    self.pipeline = FN2.CpuPacketPipeline()
            self._print('PacketPipeline: ' + type(self.pipeline).__name__)

            # b: Create and set logger
            self.logger = FN2.createConsoleLogger(FN2.LoggerLevel.NONE)
            FN2.setGlobalLogger(self.logger)

            # c: Identify device and create listener
            self.fn = FN2.Freenect2()
            serial = self.fn.getDeviceSerialNumber(0)
            self.K2device = self.fn.openDevice(serial, pipeline=self.pipeline)

            self.listener = FN2.SyncMultiFrameListener(
                FN2.FrameType.Color | FN2.FrameType.Depth)

            # d: Register listeners
            self.K2device.setColorFrameListener(self.listener)
            self.K2device.setIrAndDepthFrameListener(self.listener)

            # e: Start device and create registration
            self.K2device.start()
            self.registration = FN2.Registration(self.K2device.getIrCameraParams(), self.K2device.getColorCameraParams())

    def create_ROI(self):
   
        # a: Grab color and depth frames and register them
        reg_image = self._return_reg_color()
        #b: Select ROI using open CV
        self.r = cv2.selectROI('Image', reg_image)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        reg_image = reg_image.copy()
        # c: Save file with background rectangle
        cv2.rectangle(reg_image, (self.r[0], self.r[1]), (self.r[0] + self.r[2], self.r[1]+self.r[3]) , (0, 255, 0), 2)
        cv2.imwrite(self.master_directory+'BoundingBox.jpg', reg_image)

        self._print('ROI: Bounding box created, Image: BoundingBox.jpg, Shape: ' + str(self.r))

    def diagnose_speed(self, time = 10):
        print('Diagnosing speed for ' + str(time) + ' seconds.', file = sys.stderr)
        delta = datetime.timedelta(seconds = time)
        start_t = datetime.datetime.now()
        counter = 0
        while True:
            depth = self._return_depth()
            data = depth[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
            counter += 1
            if datetime.datetime.now() - start_t > delta:
                break
        self._print('DiagnoseSpeed: Captured ' + str(counter) + ' frames in ' + str(time) + ' seconds.')

    def create_background(self, num_frames = 5, save = True):
        print('Capturing Background', file = sys.stderr)
        self.background_time = datetime.datetime.now()
        background_data = np.empty(shape = (5, self.r[3], self.r[2]))
        background_data[:] = np.NAN
        for i in range(0,num_frames):
            background_data[i] = self.capture_frame(time = 20, save = False)
        self.background = np.nanmedian(background_data, axis = 0)
        std = np.nanstd(background_data, axis = 0)
        self.background[std > self.stdev_threshold] = np.nan
        
        if save:
            self._print('BackgroundCaptured: Background_' + str(self.background_counter).zfill(4) + '.npy, ' + str(self.background_time) + ', Med: '+ '%.2f' % np.nanmean(self.background) + ', Std: ' + '%.2f' % np.nanmean(std) + ', GP: ' + str(np.count_nonzero(~np.isnan(self.background)))  + ' of ' +  str(self.background.shape[0]*self.background.shape[1]))
            np.save(self.master_directory + 'Background_' + str(self.background_counter).zfill(4) + '.npy', self.background)
            self.background_counter += 1

    def capture_frame(self, time = 60, delta = None, save = True, background = False):
        if delta is None:
            delta= time/200

        total_time = datetime.timedelta(seconds = time)

        if datetime.datetime.now() - self.background_time > self.max_background_time:
            self.create_background()

        #Create array to hold data
        all_data = np.empty(shape = (int(time/delta), self.r[3], self.r[2]))
        all_data[:] = np.nan
        
        counter = 1
        #Collect data
        # For each received frame...
        start_t = datetime.datetime.now()
        current_delta = datetime.timedelta(seconds = counter*delta)
        
        while True:
            current_time = datetime.datetime.now()
            depth = self._return_depth()
            if (current_time - start_t) >= current_delta:
                if counter == 1: #Ignore first set of data
                    counter += 1
                    continue
                else:
                    data = depth[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
                    all_data[counter-1] = data
                    counter += 1
                    current_delta =  datetime.timedelta(seconds = counter*delta)
                    if (current_time - start_t) > total_time:
                        color = self._return_reg_color()[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]                        
                        break

        med = np.nanmedian(all_data, axis = 0)
        std = np.nanstd(all_data, axis = 0)
        med[std > self.stdev_threshold] = np.nan
        
        counts = np.count_nonzero(~np.isnan(all_data), axis = 0)
        med[counts < 5] = np.nan
        if save:
            self._print('FrameCaptured: Frame_' + str(self.frame_counter).zfill(4) + '.npy, ' + str(start_t)  + ', NFrames: ' + str(counter) + ', Med: '+ '%.2f' % np.nanmean(med) + ', Std: ' + '%.2f' % np.nanmean(std) + ', Min: ' + '%.2f' % np.nanmin(med) + ', Max: ' + '%.2f' % np.nanmax(med) + ', GP: ' + str(np.count_nonzero(~np.isnan(med)))  + ' of ' +  str(med.shape[0]*med.shape[1]))
            np.save(self.master_directory +'Frame_' + str(self.frame_counter).zfill(4) + '.npy', med)
            cv2.imwrite(self.master_directory+'Frame_' + str(self.frame_counter).zfill(4) + '.jpg', color)
            self.frame_counter += 1
        return med

    def capture_frames(self, total_time = 60*60*24*1/24):
        
        delta = datetime.timedelta(seconds = total_time)

        while True:
            now = datetime.datetime.now()
            if now - self.master_start > delta:
                break
            self.capture_frame()

            if self.PiCamera:
                if self._video_recording() and not self.camera.recording:
                    self.camera.start_recording(str((now - self.master_start).days + 1) + "_vid.h264", bitrate=7500000)
                    self._print('PiCameraStarted: Time=' + str(now) + ', File=' + str((now - self.master_start).days + 1) + "_vid.h264")
                if not self.video_recording() and self.camera.recording:
                    camera.stop_recording()
                    self._print('PiCameraStopped: Time=' + str(now) + ', File=' + str((now - self.master_start).days + 1) + "_vid.h264")

            
            
