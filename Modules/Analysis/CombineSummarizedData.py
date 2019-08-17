import glob, os
from openpyxl import load_workbook
import pandas as pd
import numpy as np

localTemporaryDirectory = os.getenv('HOME')+"/Temp/"
os.chdir(localTemporaryDirectory)

all_total = pd.DataFrame()
all_hourly = pd.DataFrame()
all_daily = pd.DataFrame()
all_davg = pd.DataFrame()
all_havg = pd.DataFrame()

for f in glob.glob(localTemporaryDirectory+'*/summarizedData_avgs.xlsx'):
	df_total=pd.read_excel(f, sheet_name="Total", index_col=0)
	all_total = all_total.append(df_total, ignore_index=True)

	df_hourly=pd.read_excel(f, sheet_name="Hourly", index_col=0)
	all_hourly=all_hourly.append(df_hourly, ignore_index=True)

	df_daily=pd.read_excel(f, sheet_name="Daily", index_col=0)
	all_daily=all_daily.append(df_daily, ignore_index=True)

	df_davg=pd.read_excel(f, sheet_name="Daily Averages", index_col=0)
	all_davg=all_davg.append(df_davg, ignore_index=True)

	df_havg=pd.read_excel(f, sheet_name="Hourly Averages", index_col=0)
	all_havg=all_havg.append(df_havg, ignore_index=True)

with pd.ExcelWriter("masterSummarizedData.xlsx") as writer:
	all_total.to_excel(writer, sheet_name="Total")
	all_daily.to_excel(writer, sheet_name="Daily")
	all_hourly.to_excel(writer, sheet_name="Hourly")
	all_davg.to_excel(writer, sheet_name="Daily Averages")
	all_havg.to_excel(writer, sheet_name="Hourly Averages")