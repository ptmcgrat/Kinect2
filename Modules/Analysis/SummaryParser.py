import argparse, glob, os, subprocess
from openpyxl import load_workbook
import pandas as pd
import numpy as np

import FileManager as FM
import DepthStats as DS

cloudMasterDirectory = 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/'
localTemporaryDirectory = os.getenv('HOME') + '/Temp/'

class SummaryParser:
	def __init__(self, methods):
		self.cloudMasterDirectory=cloudMasterDirectory
		self.localTemporaryDirectory=localTemporaryDirectory
		self.summarizedData=FM.FileManager(methods, ['summarizedData.xlsx'],
			self.cloudMasterDirectory, self.localTemporaryDirectory)
		self.fileList=self.summarizedData.fileList

		self.all_total = pd.DataFrame()
		self.all_hourly = pd.DataFrame()
		self.all_daily = pd.DataFrame()
		self.all_davg = pd.DataFrame()
		self.all_havg = pd.DataFrame()

		for i in self.fileList[0]:
			data=self.addAverages(i)
			self.combineData(data)


	def addAverages(self, file):
		#takes mean of a given metric, ignoring 0 and NaN values
		def __calculateAverage(column):
			mean=column.replace(to_replace=0, value=np.nan).mean(skipna=True)
			return mean
		
		in_file=self.localTemporaryDirectory+file

		#reads in daily, hourly, and total summaries as dataframes
		ddf=pd.read_excel(in_file, sheet_name="Daily", index_col=0)
		hdf=pd.read_excel(in_file, sheet_name="Hourly", index_col=0)
		tdf=pd.read_excel(in_file, sheet_name="Total", index_col=0)
		
		#for comparing trials of same individual
		#counts the number of hours where building was above threshhold
		hours=hdf['bowerIndex'].count()
		tdf.insert(loc=len(tdf.columns), column="thresholdHours", value=hours)
		
		columns=['castleArea', 'pitArea', 'totalArea', 'castleVolume',
		       'pitVolume', 'totalVolume', 'bowerIndex', 'bowerIndex_0.2',
		       'bowerIndex_0.4', 'bowerIndex_0.8', 'bowerIndex_1.2']
		
		#creates new dataframe for daily and hourly averages
		ddata={'projectID': ddf['projectID'][1]}
		hdata={'projectID': hdf['projectID'][1]}
		d_avg=pd.DataFrame([ddata])
		h_avg=pd.DataFrame([hdata])
		
		#takes mean of each metric and adds a column to averages DF
		for i in columns:
			if i in ddf.columns[1:]:
				d_avg[i]=__calculateAverage(ddf[i])
			if i in hdf.columns[1:]:
				h_avg[i]=__calculateAverage(hdf[i])
		
		return tdf, hdf, ddf, h_avg, d_avg
	
	def combineData(self, data):
		df_total=data[0]
		df_hourly=data[1]
		df_daily=data[2]
		df_davg=data[3]
		df_havg=data[4]

		self.all_total=self.all_total.append(df_total, ignore_index=True)
		self.all_hourly=self.all_hourly.append(df_hourly, ignore_index=True)
		self.all_daily=self.all_daily.append(df_daily, ignore_index=True)
		self.all_davg=self.all_davg.append(df_davg, ignore_index=True)
		self.all_havg=self.all_havg.append(df_havg, ignore_index=True)
