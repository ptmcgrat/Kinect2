import platform, sys, os, shutil, datetime, subprocess, smtplib, gspread, time, socket
import matplotlib.image
import Modules.LogParser as LP
import numpy as np
from PIL import Image
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from oauth2client.service_account import ServiceAccountCredentials


class CichlidTracker:
    def __init__(self):
        # 1: Define valid commands and ignore warnings
        self.commands = ['New', 'Restart', 'Stop', 'Rewrite', 'Snapshot']
        np.seterr(invalid='ignore')

        # 2: Make connection to google doc and dropbox
        self.dropboxScript = os.path.expanduser('~') + '/home/pi/Dropbox-Uploader/dropbox_uploader.sh'
        self.credentialFile = os.path.expanduser('~') + '/home/pi/SAcredentials.json'
        self._authenticateGoogleDrive()
        
        # 3: Determine which system this code is running on
        if platform.node() == 'odroid':
            self.system = 'odroid'
        elif platform.node() == 'raspberrypi' or 'Pi' in platform.node():
            self.system = 'pi'
        elif platform.system() == 'Darwin':
            self.system = 'mac'
            self.caff = subprocess.Popen('caffeinate') #Prevent mac from falling asleep
        else:
            self._initError('Could not determine which system this code is running from')

        # 4: Determine which Kinect is attached
        self._identifyDevice() #Stored in self.device
        
        # 5: Determine if PiCamera is attached
        self.piCamera = False
        if self.system == 'pi':
            from picamera import PiCamera
            self.camera = PiCamera()
            self.camera.resolution = (1296, 972)
            self.camera.framerate = 30
            self.piCamera = True
            
        # 6: Determine master directory
        self._identifyMasterDirectory() # Stored in self.masterDirectory
        
        # 7: Await instructions
        self.monitorCommands()
        
    def __del__(self):
        self._modifyPiGS(command = 'None', status = 'Stopped', error = 'UnknownError')
        self._closeFiles()
        self._uploadFiles()

    def monitorCommands(self, delta = 1):
        while True:
            self._identifyTank() #Stored in self.tankID
            command, projectID = self._returnCommand()
            if projectID in ['','None']:
                self._reinstructError('ProjectID must be set')

            print(command + '\t' + projectID)
            if command != 'None':
                self.runCommand(command, projectID)
            self._modifyPiGS(status = 'AwaitingCommand', error = '')
            time.sleep(delta*10)

    def runCommand(self, command, projectID):
        if command not in self.commands:
            self._reinstructError(command + ' is not a valid command. Options are ' + str(self.commands))
            
        if command == 'Stop':
            self._modifyPiGS(command = 'None', status = 'AwaitingCommand')
            self._closeFiles()
            return

        self._modifyPiGS(command = 'None', status = 'Running', error = '')

        self.projectID = projectID
        self.projectDirectory = self.masterDirectory + projectID + '/'
        self.loggerFile = self.projectDirectory + 'Logfile.txt'
        self.frameDirectory = self.projectDirectory + 'Frames/'
        self.backgroundDirectory = self.projectDirectory + 'Backgrounds/'
        self.videoDirectory = self.projectDirectory + 'Videos/'
        if command == 'New':
            # Project Directory should not exist. If it does, report error
            if os.path.exists(self.projectDirectory):
                self._reinstructError('New command cannot be run if ouput directory already exists. Use Rewrite or Restart')

        if command == 'Rewrite':
            if os.path.exists(self.projectDirectory):
                shutil.rmtree(self.projectDirectory)
        
        if command in ['New','Rewrite']:
            self.masterStart = datetime.datetime.now()
            os.mkdir(self.projectDirectory)
            os.mkdir(self.frameDirectory)
            os.mkdir(self.backgroundDirectory)
            os.mkdir(self.videoDirectory)
            self.frameCounter = 1
            self.backgroundCounter = 1
            self.videoCounter = 1

        if command == 'Restart':
            logObj = LP.LogParser(self.loggerFile)
            self.masterStart = logObj.master_start
            self.r = logObj.bounding_shape
            self.frameCounter = logObj.lastFrameCounter + 1
            self.backgroundCounter = logObj.lastFrameCounter + 1
            self.videoCounter = logObj.lastVideoCounter + 1
            if self.system != logObj.system or self.device != logObj.device or self.camera != logObj.camera:
                self._reinstructError('Restart error. System, device, or camera does not match what is in logfile')

        self.lf = open(self.loggerFile, 'a')
        self._modifyPiGS(start = str(self.masterStart))

        if command in ['New', 'Rewrite']:
            self._print('MasterStart: System: '+self.system + ',,Device: ' + self.device + ',,Camera: ' + str(self.piCamera) + ',,Uname: ' + str(platform.uname()))
            self._print('MasterRecordInitialStart: Time: ' + str(self.masterStart))
            self._createROI(useROI = False)

        else:
            self._print('MasterRecordRestart: Time: ' + str(datetime.datetime.now()))

            
        # Start kinect
        self._start_kinect()
        
        # Diagnose speed
        self._diagnose_speed()

        # Capture data
        self.captureFrames()
    
    def captureFrames(self, frame_delta = 5, background_delta = 60, max_frames = 100, stdev_threshold = 20):

        current_background_time = datetime.datetime.now()
        current_frame_time = current_background_time + datetime.timedelta(seconds = 60 * frame_delta)

        while True:
            # Grab new time
            now = datetime.datetime.now()

            # Fix camera if it needs to be
            if self.piCamera:
                if self._video_recording() and not self.camera.recording:
                    self.camera.start_recording(self.videoDirectory + str(self.videoCounter).zfill(4) + "_vid.h264", bitrate=7500000)
                    self._print('PiCameraStarted: Time: ' + str(datetime.datetime.now()) + ',, File: Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")
                elif not self._video_recording() and self.camera.recording:
                    self.camera.stop_recording()
                    self._print('PiCameraStopped: Time: ' + str(datetime.datetime.now()) + ',, File: Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")
                    self.videoCounter += 1
                    self._uploadFiles()

            # Capture a frame and background if necessary
            if now > current_background_time:
                out = self._captureFrame(current_frame_time, new_background = True, max_frames = max_frames, stdev_threshold = stdev_threshold)
                if out is not None:
                    current_background_time += datetime.timedelta(seconds = 60 * background_delta)
            else:
                out = self._captureFrame(current_frame_time, new_background = False, max_frames = max_frames, stdev_threshold = stdev_threshold)
            current_frame_time += datetime.timedelta(seconds = 60 * frame_delta)

            # Check google doc to determine if recording has changed.
            command, projectID = self._returnCommand()
            if command != 'None':
                self.runCommand(command, projectID)
            else:
                self._modifyPiGS(error = '')


    def _authenticateGoogleDrive(self):
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credentialFile, scope)
        gs = gspread.authorize(credentials)
        self.controllerGS = gs.open('Controller')
        pi_ws = self.controllerGS.worksheet('RaspberryPi')
        headers = pi_ws.row_values(1)
        column = headers.index('RaspberryPiID') + 1
        try:
            pi_ws.col_values(column).index(platform.node())
        except ValueError:
            pi_ws.append_row([platform.node(),socket.gethostbyname(socket.gethostname()),'','','','','','None','Stopped','Awaiting assignment of TankID',str(datetime.datetime.now())])
        
    def _identifyDevice(self):
        try:
            global freenect
            import freenect
            a = freenect.init()
            if freenect.num_devices(a) == 0:
                kinect = False
            elif freenect.num_devices(a) > 1:
                self._initError('Multiple Kinect1s attached. Unsure how to handle')
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
                self._initError('Multiple Kinect2s attached. Unsure how to handle')
            else:
                kinect2 = False
        except ImportError:
            kinect2 = False

        if kinect and kinect2:
            self._initError('Kinect1 and Kinect2 attached. Unsure how to handle')
        elif not kinect and not kinect2:
            self._initError('No depth sensor  attached')
        elif kinect:
            self.device = 'kinect'
        else:
            self.device = 'kinect2'
       
    def _identifyTank(self):
        while True:
            self._authenticateGoogleDrive() # link to google drive spreadsheet stored in self.controllerGS 
            pi_ws = self.controllerGS.worksheet('RaspberryPi')
            headers = pi_ws.row_values(1)
            raPiID_col = headers.index('RaspberryPiID') + 1
            row = pi_ws.col_values(raPiID_col).index(platform.node()) + 1
            col = headers.index('TankID')
            if pi_ws.row_values(row)[col] not in ['None','']:
                self.tankID = pi_ws.row_values(row)[col]
                self._modifyPiGS(capability = 'Device=' + self.device + ',Camera=' + str(self.piCamera), status = 'AwaitingCommand')
                return
            else:
                self._modifyPiGS(error = 'Awaiting assignment of TankID')
                time.sleep(5)

    def _identifyMasterDirectory(self):
        if self.system == 'pi':
            possibleDirs = []
            for d in os.listdir('/media/pi/'):
                try:
                    with open('/media/pi/' + d + '/temp.txt', 'w') as f:
                        print('Test', file = f)
                    with open('/media/pi/' + d + '/temp.txt', 'r') as f:
                        for line in f:
                            if 'Test' in line:
                                possibleDirs.append(d)
                except:
                    pass
                try:
                    os.remove('/media/pi/' + d + '/temp.txt')
                except FileNotFoundError:
                    continue
            if len(possibleDirs) == 1:
                self.masterDirectory = '/media/pi/' + d
            else:
                self._initError('Not sure which directory to write to. Options are: ' + str(possibleDirs))
        else:
            self.masterDirectory = 'blah'
        if self.masterDirectory[-1] != '/':
            self.masterDirectory += '/'
        if not os.path.exists(self.masterDirectory):
            os.mkdir(self.masterDirectory)
        
    def _initError(self, message):
        try:
            self._modifyPiGS(command = 'None', status = 'Stopped', error = 'InitError: ' + message)
        except:
            pass
        self._print('InitError: ' + message)
        raise TypeError
            
    def _reinstructError(self, message):
        self._modifyPiGS(command = 'None', status = 'AwaitingCommands', error = 'InstructError: ' + message)

        # Update google doc to indicate error
        self.monitorCommands()
 
    def _print(self, text):
        try:
            print(text, file = self.lf, flush = True)
        except:
            pass
        print(text, file = sys.stderr, flush = True)

    def _returnRegColor(self, crop = True):
        # This function returns a registered color array
        if self.device == 'kinect':
            out = freenect.sync_get_video()[0]
            
        elif self.device == 'kinect2':
            undistorted = FN2.Frame(512, 424, 4)
            registered = FN2.Frame(512, 424, 4)
            frames = self.listener.waitForNewFrame()
            color = frames["color"]
            depth = frames["depth"]
            self.registration.apply(color, depth, undistorted, registered, enable_filter=False)
            reg_image =  registered.asarray(np.uint8)[:,:,0:3].copy()
            self.listener.release(frames)
            out = reg_image

        if crop:
            return out[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
        else:
            return out
            
    def _returnDepth(self):
        # This function returns a float64 npy array containing one frame of data with all bad data as NaNs
        if self.device == 'kinect':
            data = freenect.sync_get_depth()[0].astype('Float64')
            data[data == 2047] = np.nan # 2047 indicates bad data from Kinect 
            return data[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]
        
        elif self.device == 'kinect2':
            frames = self.listener.waitForNewFrame(timeout = 1000)
            output = frames['depth'].asarray()
            self.listener.release(frames)
            return output[self.r[1]:self.r[1]+self.r[3], self.r[0]:self.r[0]+self.r[2]]

    def _returnCommand(self):
        self._authenticateGoogleDrive() # link to google drive spreadsheet stored in self.controllerGS 
        pi_ws = self.controllerGS.worksheet('RaspberryPi')
        headers = pi_ws.row_values(1)
        piIndex = pi_ws.col_values(headers.index('RaspberryPiID') + 1).index(platform.node())
        command = pi_ws.col_values(headers.index('Command') + 1)[piIndex]
        projectID = pi_ws.col_values(headers.index('ProjectID') + 1)[piIndex]
        return command, projectID

    def _modifyPiGS(self, start = None, command = None, status = None, IP = None, capability = None, error = None):
        self._authenticateGoogleDrive() # link to google drive spreadsheet stored in self.controllerGS 
        pi_ws = self.controllerGS.worksheet('RaspberryPi')
        headers = pi_ws.row_values(1)
        row = pi_ws.col_values(headers.index('RaspberryPiID')+1).index(platform.node()) + 1
        if start is not None:
            column = headers.index('MasterStart') + 1
            pi_ws.update_cell(row, column, start)
        if command is not None:
            column = headers.index('Command') + 1
            pi_ws.update_cell(row, column, command)
        if status is not None:
            column = headers.index('Status') + 1
            pi_ws.update_cell(row, column, status)
        if error is not None:
            column = headers.index('Error') + 1
            pi_ws.update_cell(row, column, error)
        if IP is not None:
            column = headers.index('IP')+1
            pi_ws.update_cell(row, column, IP)
        if capability is not None:
            column = headers.index('Capability')+1
            pi_ws.update_cell(row, column, capability)
        column = headers.index('Ping') + 1
        pi_ws.update_cell(row, column, str(datetime.datetime.now()))

    def _video_recording(self):
        if datetime.datetime.now().hour >= 8 and datetime.datetime.now().hour <= 12:
            return True
        else:
            return False
            
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

    def _createROI(self, useROI = False):

        # a: Grab color and depth frames and register them
        reg_image = self._returnRegColor(crop = False)
        #b: Select ROI using open CV
        if useROI:
            cv2.imshow('Image', reg_image)
            self.r = cv2.selectROI('Image', reg_image, fromCenter = False)
            self.r = tuple([int(x) for x in self.r]) # sometimes a float is returned
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            reg_image = reg_image.copy()
            # c: Save file with background rectangle
            cv2.rectangle(reg_image, (self.r[0], self.r[1]), (self.r[0] + self.r[2], self.r[1]+self.r[3]) , (0, 255, 0), 2)
            cv2.imwrite(self.master_directory+'BoundingBox.jpg', reg_image)

            self._print('ROI: Bounding box created,, Image: BoundingBox.jpg,, Shape: ' + str(self.r))
        else:
            self.r = (0,0,reg_image.shape[1],reg_image.shape[0])
            self._print('ROI: No Bounding box created,, Image: None,, Shape: ' + str(self.r))

            
    def _diagnose_speed(self, time = 10):
        print('Diagnosing speed for ' + str(time) + ' seconds.', file = sys.stderr)
        delta = datetime.timedelta(seconds = time)
        start_t = datetime.datetime.now()
        counter = 0
        while True:
            depth = self._returnDepth()
            counter += 1
            if datetime.datetime.now() - start_t > delta:
                break
        self._print('DiagnoseSpeed: Rate: ' + str(counter/time))

    def _email_summary(self):

        current_day = datetime.datetime.now().day - self.master_start.day + 1
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
        
    def _captureFrame(self, endtime, new_background = False, max_frames = 100, stdev_threshold = 20):
        # Captures time averaged frame of depth data
        
        sums = np.zeros(shape = (self.r[3], self.r[2]))
        n = np.zeros(shape = (self.r[3], self.r[2]))
        stds = np.zeros(shape = (self.r[3], self.r[2]))
        
        current_time = datetime.datetime.now()
        if current_time >= endtime:
            return
        
        while True:
            all_data = np.empty(shape = (int(max_frames), self.r[3], self.r[2]))
            for i in range(0, max_frames):
                all_data[i] = self._returnDepth()

                current_time = datetime.datetime.now()
                if current_time >= endtime:
                    break
            
            med = np.nanmedian(all_data, axis = 0)
            std = np.nanstd(all_data, axis = 0)
            med[std > stdev_threshold] = 0
            std[std > stdev_threshold] = 0
        
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
        color = self._returnRegColor()                        
        num_frames = int(max_frames*(n.max()-1) + i)
        
        self._print('FrameCaptured: NpyFile: Frames/Frame_' + str(self.frameCounter).zfill(6) + '.npy,,PicFile: Frames/Frame_' + str(self.frameCounter).zfill(6) + '.jpg,,Time: ' + str(endtime)  + ',,NFrames: ' + num_frames + ',,AvgMed: '+ '%.2f' % np.nanmean(avg_med) + ',,AvgStd: ' + '%.2f' % np.nanmean(avg_std) + ',,GP: ' + str(np.count_nonzero(~np.isnan(avg_med))))
        
        np.save(self.projectDirectory +'Frames/Frame_' + str(self.frameCounter).zfill(6) + '.npy', avg_med)
        matplotlib.image.imsave(self.projectDirectory+'Frames/Frame_' + str(self.frameCounter).zfill(6) + '.jpg', color)
        self.frameCounter += 1
        if new_background:
            self._print('BackgroundCaptured: NpyFile: Backgrounds/Background_' + str(self.backgroundCounter).zfill(6) + '.npy,,PicFile: Backgrounds/Background_' + str(self.backgroundCounter).zfill(6) + '.jpg,,Time: ' + str(endtime)  + ',,NFrames: ' + num_frames + ',,AvgMed: '+ '%.2f' % np.nanmean(avg_med) + ',,AvgStd: ' + '%.2f' % np.nanmean(avg_std) + ',,GP: ' + str(np.count_nonzero(~np.isnan(avg_med))))
            np.save(self.projectDirectory +'Backgrounds/Background_' + str(self.backgroundCounter).zfill(6) + '.npy', avg_med)
            matplotlib.image.imsave(self.projectDirectory+'Background/Background_' + str(self.backgroundCounter).zfill(6) + '.jpg', color)
            self.backgroundCounter += 1

        return avg_med

    def _uploadFiles(self):
        subprocess.Popen([self.dropboxScript, 'upload', '-s', self.projectDirectory, self.projectID]) 
    
    def _closeFiles(self):
        try:
            self._modifyPiGS(command = 'None', status = 'Stopped', error = 'ExitError: Something went wrong')
        except:
            pass
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
        if self.piCamera:
            if self.camera.recording:
                self.camera.stop_recording()
                self._print('PiCameraStopped: Time=' + str(datetime.datetime.now()) + ', File=Videos/' + str(self.videoCounter).zfill(4) + "_vid.h264")
        try:
            if self.system == 'mac':
                self.caff.kill()
        except AttributeError:
            pass

