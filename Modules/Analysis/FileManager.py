##
# command-line usage: python FileManager.py -p Projects -q Filename [Filename Filename ...] [options]
# example: python FileManager.py MethodsData.xlsx Logfile.txt
# 
# Options:
# #-r --run Execute command on each file individually after downloading
# #-c --cleanup delete all downloaded files at the end of all processes
##

import argparse, subprocess, os, glob
import pandas as pd

parser = argparse.ArgumentParser(usage='python FileManager.py Projects Filename'+
	'\n\t\t[Filename Filename...] [-i] [-r command] [-c] [-u]')
parser.add_argument('-p', '--projectFile', type=str, help='File containing list of projects to search')
parser.add_argument('-q', '--query', type=str, nargs="+", help='Name of files to download')
parser.add_argument('-c', '--cleanup', action="store_true", 
	help="delete downloaded files at the end (for use with -r)")
parser.add_argument('-r', '--run', nargs="*", 
	help="run a command on each downloaded. Uses same file order")
parser.add_argument('-i', '--include', action="store_true", help="only donwload INCLUDE=TRUE files")
parser.add_argument('-u', '--upload', action="store_true", help="upload file output from command")
args = parser.parse_args()

#ATTN: YOUR DROPBOX PATH MAY BE cichlidVideo:McGrath/...
cloudMasterDirectory = 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/'
localTemporaryDirectory = os.getenv('HOME') + '/Temp/'

class FileManager:
	def __init__(self, projectFile, query, cloudDir, localDir,
		include=True, cmd='', upload=False, cleanup=False):
		
		self.cloudMasterDirectory=cloudDir
		self.localTemporaryDirectory=localDir

		self.fileList=[] #list of files downloaded
		self.projIndex={} #project indeces for pulling files from fileList
		self.query=query
		self.include=include
		self.cmd=cmd
		self.upload=upload
		self.cleanup=cleanup

		for i in self.query: #adds dimension for each file specified in args
			self.fileList.append([])

		self.projects=pd.read_excel(projectFile, header=1)
		self.idx=0
		self.projectList=[]

		for i, row in self.projects.iterrows():
			projectID=row[0]
			methodsInclude=row[2]

			if self.include and not methodsInclude:
				pass

			else:
				self.projectList.append(projectID)

		for projectID in self.projectList:		
			print("Downloading: [" + ', '.join(self.query) + "] from " +cloudMasterDirectory+projectID)
			self.projIndex.update({projectID:self.idx})
			self.idx+=1
			
			pos=0
			for file in self.query:
				self.download(projectID, file, pos)
				pos+=1

				if self.cmd:
						run(projectID, cmd)
		
				if self.cleanup:
					cleanup(projectID)
		
				if self.upload:
					upload(projectID)

	def download(self, projectID, filename, pos): #downloads ONE FILE TYPE from ONE PROJECT
		cloudProjectDirectory=self.cloudMasterDirectory+projectID+'/'
		filePaths=self.getPaths(projectID, filename)
		
		for i in filePaths:
			name=i.split("/")[:-1] #if file is in project subdir, name should not = projectID
			localFilePath=projectID+'/'+filename
			
			if len(filePaths)>1: #in case of identical filenames within project subdirectories
				localFilePath=projectID+'/'+name+'/'+filename
			
			absLocalPath=self.localTemporaryDirectory+localFilePath
			subprocess.call(['rclone', 'copyto', cloudProjectDirectory + i, absLocalPath])
			self.fileList[pos].append(localFilePath)
	
	def getPaths(self, projectID, filename):
		cloudProjectDirectory=cloudMasterDirectory+projectID+'/'
		output=subprocess.check_output(('rclone', 'ls', cloudProjectDirectory, '--include', filename))
		result=output.decode("UTF8").strip().split('\n')
		result=[i.strip().split(' ',1) for i in result]
		filePaths=[]
		
		for i in result:
			if not i[0]:
				print('File not found... Skipping')
				self.projIndex.pop(projectID)
				self.idx-=1
				pass
			else:
				filePaths.append(i[1])
		
		return filePaths
	
	def run(self, projectID, cmd):		
		if projectID in self.projIndex:
			thisIndex=self.projIndex[projectID]
		
		else:
			return
		
		for i in self.fileList:
			cmd.insert(self.fileList.index(i)+2, self.localTemporaryDirectory + i[thisIndex])
		
		print("Running: " + ' '.join(cmd))
		output=subprocess.check_output(cmd)
	
		with open('output.csv', 'a+') as file:
			file.write(projectID + '\t' + output.decode('UTF-8'))
		
		for i in self.fileList:
			cmd.pop(2)
	
	def cleanup(self, projectID):
		if projectID in self.projIndex:
			thisIndex=self.projIndex[projectID]
	
			for i in self.fileList:
				print("Deleting " + self.localTemporaryDirectory + i[thisIndex])
				os.remove(self.localTemporaryDirectory + i[thisIndex])
	
			if not os.listdir(self.localTemporaryDirectory+projectID):
				print("Deleting empty directory " + self.localTemporaryDirectory + projectID + "/")
				os.rmdir(localTemporaryDirectory+projectID)
	
	def upload(projectID):
		pass

if args.projectFile and args.query:
	FileManager(args.projectFile, args.query,
		cloudMasterDirectory, localTemporaryDirectory, include=args.include,
		cleanup=args.cleanup,cmd=args.run, upload=args.upload)