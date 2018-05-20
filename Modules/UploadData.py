import subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('projectDirectory', type = str, help = 'Local location of data')
parser.add_argument('projectID', type = str, help = 'Folder you would like to store data in on Dropbox')
args = parser.parse_args()

dropboxScript = '/home/pi/Dropbox-Uploader/dropbox_uploader.sh'

#Log
dropbox_command = [dropboxScript, '-f', '/home/pi/.dropbox_uploader', 'upload', args.projectDirectory + '/Logfile.txt', args.projectID] 
while True:
    subprocess.call(dropbox_command, stdout = open(args.projectDirectory + 'DropboxUploadLogFile.txt', 'w'), stderr = open(args.projectDirectory + 'DropboxUploadError.txt', 'w'))
    try:
        with open(args.projectDirectory + 'DropboxUploadOutLogFile.txt') as f:
            if 'FAILED' in f.read():
                continue
            else:
                break
    except:
        break


#Backgrounds
dropbox_command = [dropboxScript, '-f', '/home/pi/.dropbox_uploader', 'upload', args.projectDirectory + '/Backgrounds', args.projectID]
while True:
    bad_flag = False
    subprocess.call(dropbox_command, stdout = open(args.projectDirectory + 'DropboxUploadBackgrounds.txt', 'w'), stderr = open(args.projectDirectory + 'DropboxUploadError.txt', 'w'))
    with open(args.projectDirectory + 'DropboxUploadBackgrounds.txt') as f:
        for line in f:
            if 'FAILED' in line:
                if 'Directory' in line:
                    continue
                else:
                    bad_flag = True
    if not bad_flag:
        break

#Frames
dropbox_command = [dropboxScript, '-f', '/home/pi/.dropbox_uploader', 'upload', args.projectDirectory + '/Frames', args.projectID]
while True:
    bad_flag = False
    subprocess.call(dropbox_command, stdout = open(args.projectDirectory + 'DropboxUploadFrames.txt', 'w'), stderr = open(args.projectDirectory + 'DropboxUploadError.txt', 'w'))
    with open(args.projectDirectory + 'DropboxUploadFrames.txt') as f:
        for line in f:
            if 'FAILED' in line:
                if 'Directory' in line:
                    continue
                else:
                    bad_flag = True
    if not bad_flag:
        break
    
#Videos
dropbox_command = [dropboxScript, '-f', '/home/pi/.dropbox_uploader', '-s', 'upload', args.projectDirectory + '/Videos', args.projectID]   
while True:
    subprocess.call(dropbox_command, stdout = open(args.projectDirectory + 'DropboxUploadVideos.txt', 'w'), stderr = open(args.projectDirectory + 'DropboxUploadError.txt', 'w'))
    errors = 0
    with open(args.projectDirectory + 'DropboxUploadVideos.txt') as f:
        for line in f:
            if 'FAILED' in line:
                if 'Directory' not in line:
                    subprocess.call([dropboxScript, '-f', '/home/pi/.dropbox_uploader', 'delete', line.split('"')[-2]])
                    errors+=1
    if errors == 0:
        break

