import argparse
import numpy as np
from scipy.ndimage.filters import uniform_filter

parser = argparse.ArgumentParser()
parser.add_argument('Infile', type = str, help = '')
parser.add_argument('Outfile', type = str, help = '')
args = parser.parse_args()

window, seconds_to_change, non_transition_bins, std, hmm_window = 120, 1800, 2, 100, 60

# Load and Nan data
ad = np.load(args.Infile)

ad[ad == 0] = 1 # 0 used for bad data to save space and use uint8 for storing data (np.nan must be a float).
	
# Calculate mean for window before and after data point (llr and rrm)
lrm = uniform_filter(ad, size = (1,window), mode = 'reflect', origin = -1*int(window/2)).astype('uint8')
rrm = np.roll(lrm, int(window), axis = 1).astype('uint8')
rrm[:,0:window] = lrm[:,0:1]

# Identify data that falls outside of mean and set it to zero
ad[(((ad > lrm + 7.5) & (ad > rrm + 7.5)) | ((ad < lrm - 7.5) & (ad < rrm - 7.5)))] = 0

np.save(args.Outfile, ad)