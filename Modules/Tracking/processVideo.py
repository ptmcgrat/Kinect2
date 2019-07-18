import argparse, subprocess, datetime
from .. import LogParser as LP

parser = argparse.ArgumentParser()
parser.add_argument('VideoFile', type = str, help = 'Name of h264 file to be processed')
parser.add_argument('LogFile', type = str, help = 'Name of logfile')
parser.add_argument('MasterDirectory', type = str, help = 'Masterdirectory')
parser.add_argument('CloudDirectory', type = str, help = 'Masterdirectory')

uploadLog = args.MasterDirectory + 'ProcessLog.txt'

args = parser.parse_args()

tol = 0.001

if '.h264' not in args.VideoFile:
	raise Exception(args.VideoFile + ' not an h264 file')

lp = LP.LogParser(self.loggerFile)
baseName = args.VideoFile.split('/')[-1]

with open(uploadLog, 'a') as f:
	print('VideoProcessStart: ' + baseName + ' - ' + str(datetime.datetime.now()), file = f)

videoObjs = [x for x in lp.movies if baseName in x.h264_file]
assert len(VideoObjs) == 1

framerate = videoObjs[0].framerate
height = videoObj[0].height
width = videoObj[0].width
predicted_frames = int((videoObj[0].end_time - videoObj[0].time).total_seconds()*framerate)

# Convert h264 to mp4
subprocess.call(['ffmpeg', '-r', str(framerate), '-i', args.VideoFile, '-threads', '1', '-c:v', 'copy', '-r', str(framerate), args.VideoFile.replace('.h264', '.mp4')])
assert os.path.isfile(args.VideoFile.replace('.h264', '.mp4'))

with open(uploadLog, 'a') as f:
	print('VideoConverted: ' + baseName + ' - ' + str(datetime.datetime.now()), file = f)

# Get stats to make sure video looks right
subprocess.call(['ffprobe', args.VideoFile.replace('.h264', '.mp4'), '-v', 'error', '-show_entries', 'stream=width,height,avg_frame_rate,duration'], stdout = open(args.MasterDirectory + 'stats.txt', 'w'))

with open(args.MasterDirectory + 'stats.txt') as f:
	for line in f:
		try:
			dataType, dataValue = line.rstrip().split('=')
			if dataType == 'width':
				new_width = int(dataValue)
			if dataType == 'height':
				new_height = int(dataValue)
			if dataType == 'avg_frame_rate':
				new_framerate = int(dataValue.split('/')[0])/int(dataValue.split('/')[1])
			if dataType == 'duration':
				new_frames = int(int(dataValue)*new_framerate)
		except IndexError:
			continue

assert new_height == height
assert new_width == width
assert abs(new_framerate - framerate) < tol*framerate
assert abs(predicted_frames - new_frames) < tol*predicted_frames

with open(uploadLog, 'a') as f:
	print('VideoValidated: ' + baseName + ' - ' + str(datetime.datetime.now()), file = f)


# Sync with cloud (will return error if something goes wrong)
subprocess.call(['rclone', 'copy', args.VideoFile.replace('.h264', '.mp4'), args.CloudDirectory], check = True)

with open(uploadLog, 'a') as f:
	print('VideoUploaded: ' + baseName + ' - ' + str(datetime.datetime.now()), file = f)


# Delete videos
subprocess.call(['rm', '-f', args.VideoFile, args.VideoFile.replace('.h264', '.mp4')])

with open(uploadLog, 'a') as f:
	print('VideoFinished: ' + baseName + ' - ' + str(datetime.datetime.now()), file = f)

