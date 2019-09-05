#import argparse, datetime
#from scipy.ndimage.filters import uniform_filter
#import numpy as np
#from hmmlearn import hmm
#import LogParser as LP
#np.warnings.filterwarnings('ignore')

import argparse, os, psutil
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('NpyFile', type = str, help = '')
#parser.add_argument('FrameRate', type = int, help = '')
#parser.add_argument('NumSeconds', type = int)
#parser.add_argument('Width', type = int)
#parser.add_argument('RowIndex', type = int)
args = parser.parse_args()

process = psutil.Process(os.getpid())
print(process.memory_info().rss)  # in bytes 
call(['python3', '_HMMNan.py', args.NpyFile, args.NpyFile.replace('.npy', '.nan.npy')])
print(process.memory_info().rss)  # in bytes 
call(['python3', '_HMMInterp.py', args.NpyFile.replace('.npy', '.nan.npy'), args.NpyFile.replace('.npy', '.interp.npy')])
print(process.memory_info().rss)  # in bytes 


#ad = np.load(args.NpyFile)

"""
ad = np.zeros(shape = (args.Width, args.NumSeconds), dtype = 'uint8')

cap = cv2.VideoCapture(args.VideoFile)
for i in range(args.NumSeconds):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i*args.FrameRate))
    ret, frame = cap.read()
    if not ret:
        raise Exception('Cant read frame')
    ad[:,i] =  0.2125 * frame[args.RowIndex,:,2] + 0.7154 * frame[args.RowIndex,:,1] + 0.0721 * frame[args.RowIndex,:,0] #opencv does bgr instead of rgb
cap.release()
"""

"""
# Reshape array to save it
data = np.reshape(ad, newshape = original_shape, order = 'C').astype('uint8')

print('CalculatingHMM: ' + str(datetime.datetime.now()))

outData = np.empty(shape = (50*data.shape[0], 3), dtype = int)
count = 0
for i, column in enumerate(data):
	print(i) if i%100 == 0 else None
	means = scipy.ndimage.filters.uniform_filter(column, size = hmm_window, mode = 'reflect').astype('uint8')
	freq, bins = np.histogram(means, bins = range(0,257,2))
            
	states = bins[0:-1][freq > hmm_window]
	comp = len(states)
	if comp == 0:
	    print('For row ' + str(row) + ' and column ' + str(i) + ', states = ' + str(states))
	    states = [125]
            
	model = hmm.GaussianHMM(n_components=comp, covariance_type="spherical")
	model.startprob_ = np.array(comp*[1/comp])
	change = 1/(seconds_to_change)
	trans = np.zeros(shape = (len(states),len(states)))
	for k,state in enumerate(states):
		s_trans = np.zeros(shape = states.shape)
		n_trans_states = np.count_nonzero((states > state + non_transition_bins) | (states < state - non_transition_bins))
		if n_trans_states != 0:
			s_trans[(states > state + non_transition_bins) | (states < state - non_transition_bins)] = change/n_trans_states
			s_trans[states == state] = 1 - change
		else:
			s_trans[states == state] = 1
		trans[k] = s_trans
                   
	model.transmat_ = np.array(trans)
	model.means_ = states.reshape(-1,1)
	model.covars_ = np.array(comp*[std])
            
	z = [model.means_[x][0] for x in model.predict(column.reshape(-1,1))]

	zs = np.split(z, 1 + np.where(np.diff(z) != 0)[0])

	cpos = 0
	for d in zs:
		try:
			outData[count] = (cpos, cpos + len(d) - 1, d[0])
		except IndexError:
			outData = np.resize(outData, (outData.shape[0]*5, outData.shape[1]))
			outData[count] = (cpos, cpos + len(d) - 1, d[0])
		cpos = cpos + len(d)
		count += 1

outData = np.delete(outData, range(count, outData.shape[0]), axis = 0)

print('Finished: ' + str(datetime.datetime.now()))

"""
