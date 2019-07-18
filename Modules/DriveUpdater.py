import argparse, datetime, gspread
from LogParser import LogParser as LP
import matplotlib
matplotlib.use('Pdf')  # Enables creation of pdf without needing to worry about X11 forwarding when ssh'ing into the Pi
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

parser = argparse.ArgumentParser()
parser.add_argument('Logfile', type = str, help = 'Name of logfile')
args = parser.parse_args()

class DriveUpdater:
    def __init__(self, logfile):
        self.lp = LP(logfile)
        self.node = self.lp.uname.split("node='")[1].split("'")[0]
        self.lastFrameTime = self.lp.frames[-1].time
        self.masterDirectory = logfile.split(self.lp.projectID)[0]
        self.projectDirectory = self.masterDirectory + self.lp.projectID + '/'
        self.credentialDrive = self.masterDirectory + 'CredentialFiles/DriveCredentials.txt'
        self.credentialSpreadsheet = self.masterDirectory + 'CredentialFiles/SAcredentials.json'
        self._createImage()
        f = self.uploadImage(self.lp.tankID + '.jpg', self.lp.tankID)
        self.insertImage(f)

    def _createImage(self):
        lastHourFrames = [x for x in self.lp.frames if x.time > self.lastFrameTime - datetime.timedelta(hours = 1)]  
        lastDayFrames = [x for x in self.lp.frames if x.time > self.lastFrameTime - datetime.timedelta(days = 1)]
        t_change = str(self.lastFrameTime - self.lp.frames[0].time)
        d_change = str(self.lastFrameTime - lastDayFrames[0].time)
        h_change = str(self.lastFrameTime - lastHourFrames[0].time)
        
        fig = plt.figure(figsize=(10,7))
        fig.suptitle(self.lastFrameTime)
        ax1 = fig.add_subplot(2, 3, 1) #Pic from Kinect
        ax2 = fig.add_subplot(2, 3, 2) #Pic from Camera
        ax3 = fig.add_subplot(2, 3, 3) #Depth from Kinect
        ax4 = fig.add_subplot(2, 3, 4) #Total Depth Change
        ax5 = fig.add_subplot(2, 3, 5) #Day Depth Change
        ax6 = fig.add_subplot(2, 3, 6) #Hour Depth Change

        img_1 = img.imread(self.projectDirectory + self.lp.frames[-1].pic_file)
        try:
            img_2 = img.imread(self.projectDirectory + self.lp.movies[-1].pic_file)
        except:
            img_2 = img_1
        dpth_3 = np.load(self.projectDirectory + self.lp.frames[-1].npy_file)
        dpth_4 = np.load(self.projectDirectory + self.lp.frames[0].npy_file)
        dpth_5 = np.load(self.projectDirectory + lastDayFrames[0].npy_file)
        dpth_6 = np.load(self.projectDirectory + lastHourFrames[0].npy_file)

        ### TITLES ###
        ax1.set_title('Kinect RGB Picture')
        ax2.set_title('PiCamera RGB Picture')
        ax3.set_title('Current Kinect Depth')
        ax4.set_title('Total Change\n'+t_change)
        ax5.set_title('Last 24 hours change\n'+d_change)
        ax6.set_title('Last 1 hour change\n'+h_change)
        
        ax1.imshow(img_1)
        ax2.imshow(img_2)
        ax3.imshow(dpth_3)
        ax4.imshow(dpth_4 - dpth_3, vmin = -5, vmax = 5)
        ax5.imshow(dpth_5 - dpth_3, vmin = -5, vmax = 5)
        ax6.imshow(dpth_6 - dpth_3, vmin = -5, vmax = 5)
        
        #plt.subplots_adjust(bottom = 0.15, left = 0.12, wspace = 0.24, hspace = 0.57)
        plt.savefig(self.lp.tankID + '.jpg')
        #return self.graph_summary_fname       
    
    def uploadImage(self, image_file, name): #name should have format 't###_icon' or 't###_link'
        self._authenticateGoogleDrive()
        drive = GoogleDrive(self.gauth)
        folder_id = "'151cke-0p-Kx-QjJbU45huK31YfiUs6po'"  #'Public Images' folder ID
        
        file_list = drive.ListFile({'q':"{} in parents and trashed=false".format(folder_id)}).GetList()
        # check if file name already exists so we can replace it
        flag = False
        count = 0
        while flag == False and count < len(file_list):
            if file_list[count]['title'] == name:
                fileID = file_list[count]['id']
                flag = True
            else:
                count += 1

        if flag == True:
            # Replace the file if name exists
            f = drive.CreateFile({'id': fileID})
            f.SetContentFile(image_file)
            f.Upload()
            # print("Replaced", name, "with newest version")
        else:
            # Upload the image normally if name does not exist
            f = drive.CreateFile({'title': name, 'mimeType':'image/jpeg',
                                 "parents": [{"kind": "drive#fileLink", "id": folder_id[1:-1]}]})
            f.SetContentFile(image_file)
            f.Upload()                   
            # print("Uploaded", name, "as new file")
        return f

    def insertImage(self, f):
        self._authenticateGoogleSpreadSheets()
        pi_ws = self.controllerGS.worksheet('RaspberryPi')
        headers = pi_ws.row_values(1)
        raPiID_col = headers.index('RaspberryPiID') + 1
        image_col = headers.index('Image') + 1
        row = pi_ws.col_values(raPiID_col).index(self.node) + 1
        
        info = '=HYPERLINK("' + f['alternateLink'] + '", IMAGE("' + f['webContentLink'] + '"))'

        pi_ws.update_cell(row, image_col, info)     
    
    def _authenticateGoogleSpreadSheets(self):
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credentialSpreadsheet, scope)
        try:
            gs = gspread.authorize(credentials)
            while True:
                try:
                    self.controllerGS = gs.open('Controller')
                    pi_ws = self.controllerGS.worksheet('RaspberryPi')
                    break
                except:
                    time.sleep(1)
        except gspread.exceptions.RequestError:
            time.sleep(2)
            self._authenticateGoogleSpreadSheets()

    def _authenticateGoogleDrive(self):
        self.gauth = GoogleAuth()
        # Try to load saved client credentials
        self.gauth.LoadCredentialsFile(self.credentialDrive)
        if self.gauth.credentials is None:
            # Authenticate if they're not there
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # Refresh them if token is expired
            self.gauth.Refresh()
        else:
            # Initialize with the saved creds
            self.gauth.Authorize()
        # Save the current credentials to a file
        self.gauth.SaveCredentialsFile(self.credentialDrive)

dr_obj = DriveUpdater(args.Logfile)
