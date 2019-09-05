#DOESN'T WORK... YET

import os
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

class DepthStats:
	def __init__(self, methods, data, groups=None):
		self.methods=methods
		self.data=data
		self.groups=groups
		if not self.groups:
			self.groups=self.data.groupID.unique()
		self.sampleInfo=pd.DataFrame(columns=['projectID',
			'groupID', 'individual', 'trial'])
	
		for column,row in self.methods.iterrows():
			projectID=row[0]
			groupID=row[1]
	
			if 'con' in projectID:
				self.sampleInfo=self.sampleInfo.append({'projectID': projectID, 
					'groupID': groupID}, ignore_index=True)
			else:
				indivID=[x for x in projectID[len(groupID):].split('_') if x]
				self.sampleInfo=self.sampleInfo.append({
					'projectID': projectID, 'groupID': groupID,
					'individual': indivID[0], 'trial': indivID[1]},
				ignore_index=True)

		self.data=self.data.join(self.sampleInfo.set_index('projectID'), on='projectID')
		self.data.sort_values('thresholdHours', ascending=False, inplace=True)
		self.data.drop_duplicates(subset=['groupID', 'individual'],
			keep='first', inplace=True)
		self.data.sort_values('projectID', inplace=True)
		self.metrics=['castleArea', 'pitArea', 'totalArea', 'castleVolume',
		       'pitVolume', 'totalVolume', 'bowerIndex', 'bowerIndex_0.2',
		       'bowerIndex_0.4', 'bowerIndex_0.8', 'bowerIndex_1.2']


		self.shapiro=pd.DataFrame(columns=['groupID']+self.metrics)



		self.checkNormal(self.groups)

	def checkNormal(self, testGroup):
		#return bool? return summary DF?
		for i in testGroup:
			row={'groupID': i}
			for j in self.metrics :
				x=self.data.set_index('groupID').loc[i][j]
				if x.size<=3:
					break
				else:
					w,p = stats.shapiro(x)
					row.update({j: (w,p)})
			self.shapiro=self.shapiro.append(row, ignore_index=True)

		bools={'groupID': 'all'}
		for i in self.metrics:
				val=self.shapiro.max(i)
				print(val)



	def equalVariance(self, normal=True):
		result=pd.DataFrame(columns=self.metrics)
		for i in self.metrics:
			toStat=[]
			row=[]
			for j in data.groupID.unique():
				x=data.loc[data['groupID']==j][i]
				toStat.append(x)
			if normal:
				p,w = stats.levene(*toStat)
			else:
				p,w = stats.bartlett(*toStat)
			row.append((w,p))
		result.append(row)
		return result

	def __defineGroups(self):
		pass
		#check if groups are already defined (passed by user)
		#split groups based on some other parameters

controls=['FemaleControl', 'SingleMale', 'SocialMale', 'Empty']
pit=['CV', 'TI']
castle=['MC']
f1=['MCxCVF1', 'TIxMCF1']
group1=[controls, pit, castle]
group2=[controls, pit+castle, f1]

methodsData=pd.read_excel(os.getenv('HOME')+'/Kinect2/MethodsData.xlsx', header=1, ignore_index=True)
summarizedData=pd.read_excel(os.getenv('HOME')+'/Kinect2/masterSummarizedData.xlsx', sheet_name='Total', 
	header=0, index_col=0, ignore_index=True)
DepthStats(methodsData, summarizedData, groups=group2)








