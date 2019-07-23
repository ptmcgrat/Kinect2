import argparse
import numpy as np
from scipy.ndimage.filters import uniform_filter

parser = argparse.ArgumentParser()
parser.add_argument('Infile', type = str, help = '')
parser.add_argument('Outfile', type = str, help = '')
args = parser.parse_args()


ad = np.load(args.Infile)
# Interpolation missing data for HMM
ad = ad.ravel(order = 'C') #np.interp requires flattend data
nans, x = ad==0, lambda z: z.nonzero()[0]
ad[nans]= np.interp(x(nans), x(~nans), ad[~nans])

np.save(args.Outfile, ad)