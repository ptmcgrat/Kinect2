import argparse, os
import pandas as pd
import numpy as np
from openpyxl import load_workbook

parser = argparse.ArgumentParser(usage='calculate Daily & Hourly depth change averages')
parser.add_argument('file', type=str, help='summarizedData.xlsx file')
args = parser.parse_args()

ddf=pd.read_excel(args.file, sheet_name="Daily", index_col=0)
hdf=pd.read_excel(args.file, sheet_name="Hourly", index_col=0)
tdf=pd.read_excel(args.file, sheet_name="Total", index_col=0)

tdf.insert(loc=len(tdf.columns), column="thresholdHours", value=hdf['bowerIndex'].count())

columns=['projectID', 'castleArea', 'pitArea', 'totalArea', 'castleVolume',
       'pitVolume', 'totalVolume', 'bowerIndex', 'bowerIndex_0.2',
       'bowerIndex_0.4', 'bowerIndex_0.8', 'bowerIndex_1.2']

def __calculateAverage(column):
	mean=column.replace(to_replace=0, value=np.nan).mean(skipna=True)
	return mean

ddata={'projectID': ddf['projectID'][1]}
hdata={'projectID': hdf['projectID'][1]}

d_avg=pd.DataFrame([ddata])
h_avg=pd.DataFrame([hdata])

for i in columns:
	if i in ddf.columns[1:]:
		d_avg[i]=__calculateAverage(ddf[i])
	if i in hdf.columns[1:]:
		h_avg[i]=__calculateAverage(hdf[i])

basename=os.path.basename(args.file).split('.')[0]
path=os.path.dirname(args.file)+"/"

with pd.ExcelWriter(path+basename+"_avgs.xlsx") as writer:
	tdf.to_excel(writer, sheet_name="Total")
	ddf.to_excel(writer, sheet_name="Daily")
	hdf.to_excel(writer, sheet_name="Hourly")
	d_avg.to_excel(writer, sheet_name="Daily Averages")
	h_avg.to_excel(writer, sheet_name="Hourly Averages")




