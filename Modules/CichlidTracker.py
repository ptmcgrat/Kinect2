import cv2, platform, sys, os, shutil, datetime, subprocess, smtplib
import Modules.LogParser as LP
import numpy as np
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


class CichlidTracker:
    def __init__(self, project_name, output_directory, rewrite_flag, frame_delta = 5, background_delta = 60, ROI = False):

        # 1: Set parameters
        self.project_name = project_name
        self.frame_counter = 1 # Keep track of how many frames are saved
        self.background_counter = 1 # Keep track of how many backgrounds are saved
        self.kinect_bad_data = 2047
        np.seterr(invalid='ignore')

        # 2: Determine which system this code is running on
        if platform.node() == 'odroid':
            self.system = 'odroid'
        elif platform.node() == 'raspberrypi' or 'Pi' in platform.node():
            self.system = 'pi'
        elif platform.system() == 'Darwin':
            self.system = 'mac'
            self.caff = subprocess.Popen('caffeinate') #Prevent mac from falling asleep
        else:
            print('Not sure what system you are running on.', file = sys.stderr)
            sys.exit()

        # 3: Determine which Kinect is attached
        self._identify_device() #Stored in self.device

        # 4: Determine if PiCamera is attached
        self.PiCamera = False
        if 'pi' in self.system.lower():
            from picamera import PiCamera
            self.camera = PiCamera()
            self.camera.resolution = (1296, 972)
            self.camera.framerate = 30
            self.PiCamera = True

        # 5: Create master directory and logging files
        if output_directory is None:
            if self.system == 'odroid':
                self.master_directory = '/media/odroid/Kinect2/' + project_name + '/'
            elif self.system == 'pi':
                self.master_directory = '/media/pi/Kinect2/' + project_name + '/'
            elif self.system == 'mac':
                self.master_directory = '/Users/pmcgrath7/Dropbox (GaTech)/Applications/KinectPiProject/Kinect2Tests/Output/' + project_name + '/'
        else:
            if output_directory[-1] != '/':
                output_directory += '/'
            self.master_directory = output_directory + project_name + '/'

        if rewrite_flag:
            if os.path.exists(self.master_directory):
                shutil.rmtree(self.master_directory)

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        if not os.path.exists(self.master_directory):
            os.mkdir(self.master_directory)
        else:
            print('Project directory already exists. If you would like to overwrite, use the -r flag')
            sys.exit()

        if not os.path.exists(self.master_directory + 'Frames/'):
            os.mkdir(self.master_directory + 'Frames/')

        if not os.path.exists(self.master_directory + 'Backgrounds/'):
            os.mkdir(self.master_directory + 'Backgrounds/')

        if self.PiCamera and not os.path.exists(self.master_directory + 'Videos/'):
            os.mkdir(self.master_directory + 'Videos/')
        
        self.logger_file = self.master_directory + 'Logfile.txt'
        self.lf = open(self.logger_file, 'w')
        self._print('MasterStart: System='+self.system + ',,Device=' + self.device + ',,Camera=' + str(self.PiCamera) + ',,Uname=' + str(platform.uname()))
        self._print('MasterStart: Time=' + str(datetime.datetime.now()))

        # 6: Open and start Kinect2
        self._start_kinect()

        # 7: Identify ROI for depth data study
        self._create_ROI(use_ROI = False)
            
        # 8: Diagnose speed
        self._diagnose_speed()
        
    def __del__(self):
        try:
            self._print('MasterRecordStop: ' + str(datetime.datetime.now()))
            self.lf.close()
        except AttributeError:
            pass
        try:
            if self.device == 'kinect2':
                self.K2device.stop()
            if self.device == 'kinect':
                freenect.sync_stop()
        except AttributeError:
            pass
        try:
            if self.system == 'mac':
                self.caff.kill()
        except AttributeError:
            pass
        
    def _print(self, text):
        print(text, file = self.lf)
        print(text, file = sys.stderr)

    def _return_reg_color(self):
        if self.device == 'kinect':
            return freenect.sync_get_video()[0][self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]

        elif self.device == 'kinect2':
            undistorted = FN2.Frame(512, 424, 4)
            registered = FN2.Frame(512, 424, 4)
            frames = self.listener.waitForNewFrame()
            color = frames["color"]
            depth = frames["depth"]
            self.registration.apply(color, depth, undistorted, registered, enable_filter=False)
            reg_image =  registered.asarray(np.uint8)[:,:,0:3].copy()
            self.listener.release(frames)
            return reg_image[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]

    def _return_depth(self):
        if self.device == 'kinect':
            data = freenect.sync_get_depth()[0].astype('Float64')
            data[data == self.kinect_bad_data] = np.nan
            return data[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
        
        elif self.device == 'kinect2':
            frames = self.listener.waitForNewFrame(timeout = 1000)
            output = frames['depth'].asarray()
            self.listener.release(frames)
            return output[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]

    def _video_recording(self):
        if datetime.datetime.now().hour >= 8 and datetime.datetime.now().hour <= 12
            return True
        else:
            return False
        
    def _identify_device(self):
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

    def _start_kinect(self):
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

    def _create_ROI(self, use_ROI = False):

        # a: Grab color and depth frames and register them
        reg_image = self._return_reg_color()
        #b: Select ROI using open CV
        if use_ROI:
            cv2.imshow('Image', reg_image)
            self.r = cv2.selectROI('Image', reg_image, fromCenter = False)
            self.r = tuple([int(x) for x in self.r]) # sometimes a float is returned
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            reg_image = reg_image.copy()
            # c: Save file with background rectangle
            cv2.rectangle(reg_image, (self.r[0], self.r[1]), (self.r[0] + self.r[2], self.r[1]+self.r[3]) , (0, 255, 0), 2)
            cv2.imwrite(self.master_directory+'BoundingBox.jpg', reg_image)

            self._print('ROI: Bounding box created, Image: BoundingBox.jpg, Shape: ' + str(self.r))
        else:
            self.r = (0,0,reg_image.shape[1],reg_image[0])
            self._print('ROI: No Bounding box created, Image: None, Shape: ' + str(self.r))

            
    def _diagnose_speed(self, time = 10):
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
        self._print('DiagnoseSpeed: Rate: ' + str(counter/time))

    def _email_summary(self):

        current_day = datetime.datetime.now().day - self.master_start.day + 1)
        # Create summary plot
        
        recipients = ['patrick.mcgrath@biology.gatech.edu', 'ptmcgrat@gmail.com'] # ADD EMAIL HERE
        msg = MIMEMultipart()
        msg['From'] = platform.node + '@biology.gatech.edu'
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = 'Daily summary from [' + platform.node + '] for day: ' + str(current_day)

        msgAlternative = MIMEMultipart('alternative')
        msg.attach(msgAlternative)

        msgText = MIMEText('This is the alternative plain text message.')
        msgAlternative.attach(msgText)

        # We reference the image in the IMG SRC attribute by the ID we give it below
        msgText = MIMEText('<b>Some <i>HTML</i> text</b> and an image.<br><img src="cid:image1"><br>Nifty!', 'html')
        msgAlternative.attach(msgText)

        # Get the day summary from the 
        lp = LP.LogParser(self.lf)
        msgImage = MIMEImage(lp.day_summary(current_day))
        
        # Define the image's ID as referenced above
        msgImage.add_header('Content-ID', '<image1>')
        msg.attach(msgImage)
               
        server=smtplib.SMTP('outbound.gatech.edu', 25)
        server.starttls()
        server.send_message(msg)
        server.quit()    

    def capture_frame(self, endtime, new_background = False, max_frames = 100, stdev_threshold = 20):

        sums = np.zeros(shape = (self.r[3], self.r[2]))
        n = np.zeros(shape = (self.r[3], self.r[2]))
        stds = np.zeros(shape = (self.r[3], self.r[2]))
        
        current_time = datetime.datetime.now()
        if current_time >= endtime:
            return
        
        while True:
            all_data = np.empty(shape = (int(max_frames), self.r[3], self.r[2]))
            for i in range(0, max_frames):
                all_data[i] = self._return_depth()

                current_time = datetime.datetime.now()
                if current_time >= endtime:
                    break
            
            med = np.nanmedian(all_data, axis = 0)
            std = np.nanstd(all_data, axis = 0)
            med[std > self.stdev_threshold] = 0
            std[std > self.stdev_threshold] = 0
        
            counts = np.count_nonzero(~np.isnan(all_data), axis = 0)
            med[counts < 5] = 0
            std[counts < 5] = 0
            
            sums += med
            stds += std
            med[counts > 1] = 1
            n += med

            current_time = datetime.datetime.now()
            if current_time >= endtime:
                break

        avg_med = sums/n
        avg_std = stds/n
        color = self._return_reg_color()[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]                        

        self._print('FrameCaptured: NpyFile: Frames/Frame_' + str(self.frame_counter).zfill(6) + '.npy,,PicFile: Frames/Frame_' + str(self.frame_counter).zfill(6) + '.jpg,,Time' + str(start_t)  + ',,AvgMed: '+ '%.2f' % np.nanmean(avg_med) + ',,AvgStd: ' + '%.2f' % np.nanmean(avg_std) + ',,GP: ' + str(np.count_nonzero(~np.isnan(avg_med))))
        np.save(self.master_directory +'Frames/Frame_' + str(self.frame_counter).zfill(6) + '.npy', avg_med)
        cv2.imwrite(self.master_directory+'Frames/Frame_' + str(self.frame_counter).zfill(6) + '.jpg', color)
        self.frame_counter += 1
        if new_background:
            self._print('BackgroundCaptured: NpyFile: Backgrounds/Background_' + str(self.background_counter).zfill(6) + '.npy,,PicFile: Backgrounds/Background_' + str(self.background_counter).zfill(6) + '.jpg,,Time' + str(start_t)  + ',,AvgMed: '+ '%.2f' % np.nanmean(avg_med) + ',,AvgStd: ' + '%.2f' % np.nanmean(avg_std) + ',,GP: ' + str(np.count_nonzero(~np.isnan(avg_med))))
            np.save(self.master_directory +'Backgrounds/Background_' + str(self.background_counter).zfill(6) + '.npy', avg_med)
            cv2.imwrite(self.master_directory+'Backgrounds/Background_' + str(self.background_counter).zfill(6) + '.jpg', color)
            self.background_counter += 1
            self.background = avg_med

        return avg_med

    def capture_frames(self, total_time = 60*60*24*1/24, frame_delta = 5, background_delta = 60, max_frames = 100, stdev_threshold = 20):
        
        self.master_start = datetime.datetime.now()
        total_delta = datetime.timedelta(seconds = total_time)
        frame_delta = datetime.timedelta(seconds = 60 * frame_delta)
        background_delta = datetime.timedelta(seconds = 60 * background_delta)

        current_frametime = master_start + frame_delta
        current_backgroundtime = master_start
        
        while True:
            # Grab new time
            now = datetime.datetime.now()

            # Is the recording finished?
            if now - self.master_start > total_delta:
                break

            # Fix camera if it needs to be
            if self.PiCamera:
                if self._video_recording() and not self.camera.recording:
                    self.camera.start_recording(self.master_directory + 'Videos/' + str(now.day - self.master_start.day + 1) + "_vid.h264", bitrate=7500000)
                    self._print('PiCameraStarted: Time=' + str(datetime.datetime.now()) + ', File=Videos/' + str(now.day - self.master_start.day + 1) + "_vid.h264")
                elif not self._video_recording() and self.camera.recording:
                    self.camera.stop_recording()
                    self._print('PiCameraStopped: Time=' + str(datetime.datetime.now()) + ', File=Videos/' + str(now.day - self.master_start.day + 1) + "_vid.h264")
                    self._email_summary()

            # Capture a frame and background if necessary
            if now > current_background_time:
                out = self.capture_frame(current_frametime, background = True, max_frames = max_frames, stdev_threshold = stdev_threshold)
                if out is not None:
                    current_background_time += background_delta
            else:
                out = self.capture_frame(current_frametime, background = False, max_frames = max_frames, stdev_threshold = stdev_threshold)
            current_frametime += frame_delta

            
            
