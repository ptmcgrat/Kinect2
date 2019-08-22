##
# usage: python GetFiles.py Projects Filename [Filename Filename ...] [options]
# example: python GetFiles.py MethodsData.xlsx Logfile.txt
#
# 
# Options:
# #-r --run Execute command on each file individually after downloading
# #-c --cleanup delete all downloaded files at the end of all processes
# #-i --include only process files with INCLUDE=TRUE in MethodsData
##

import argparse, subprocess, os
import pandas as pd

parser = argparse.ArgumentParser(usage='python FileGrab.py Projects Filename\n\t\t[Filename Filename...] [-r command] [-c] [-u]')
parser.add_argument('projectFile', type=str, help='File containing list of projects to search')
parser.add_argument('query', type=str, nargs="+", help='Name of files to download')
parser.add_argument('-c', '--cleanup', action="store_true", help="delete downloaded files at the end (for use with -r)")
parser.add_argument('-r', '--run', nargs="*", help="run a command on each downloaded. Uses same file order")
parser.add_argument('-i', '--include', action="store_true", help="only donwload INCLUDE=TRUE files")
args = parser.parse_args()

cloudMasterDirectory = 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/'
localTemporaryDirectory = os.getenv('HOME') + '/Temp/'

fileList=[]
projIndex={}
for i in args.query:
	fileList.append([])

def __downloadFile(projectID, filename, pos):
	cloudProjectDirectory=cloudMasterDirectory+projectID+'/'
	filePaths=__getPaths(projectID, filename)
	
	for i in filePaths:
		name=i.split("/")[0]
		localFilePath=localTemporaryDirectory+projectID+'_'+filename
		
		if len(filePaths)>1:
			localFilePath=localTemporaryDirectory+projectID+'_'+name+'_'+filename
		
		subprocess.call(['rclone', 'copyto', cloudProjectDirectory + i, localFilePath])
		fileList[pos].append(localFilePath)

def __getPaths(projectID, filename):
	cloudProjectDirectory=cloudMasterDirectory+projectID+'/'
	output=subprocess.check_output(('rclone', 'ls', cloudProjectDirectory, '--include', filename))
	result=output.decode("UTF8").strip().split('\n')
	result=[i.strip().split(' ',1) for i in result]
	filePaths=[]
	
	for i in result:
		if not i[0]:
			print('File not found... Exiting')
			#quit()
		else:
			filePaths.append(i[1])
	
	return filePaths

def __runCommand(projectID):
	cmd = args.run
	index=projIndex[projectID]
	
	for i in fileList:
		cmd.insert(fileList.index(i)+2, i[index])
	
	print("Running: " + ' '.join(cmd))
	output = subprocess.check_output(cmd)

	with open(localTemporaryDirectory+"/output.csv") as outfile:
		outfile.write(projectID + "\t" + output)
	
	for i in fileList:
		cmd.pop(2)

def __cleanup():
	for i in fileList:
		for j in i:
			print("Deleting " + j)
			os.remove(j)


projects=pd.read_excel(args.projectFile, header=1)
index=0
for index, row in projects.iterrows():
	projectID=row[0]
	include=row[2]
	pos=0

	if args.include and not include:
		pass

	else:
		print("Downloading: [" + ', '.join(args.query) + "] from " + cloudMasterDirectory+projectID)
		projIndex.update({projectID:index})
		index+=1
	
		for file in args.query:
			__downloadFile(projectID, file, pos)
			pos+=1

print('Downloads finished!')
print()

if args.run:
	for index, row in projects.iterrows():
		projectID=row[0]
		__runCommand(projectID)

if args.cleanup:
	__cleanup()
