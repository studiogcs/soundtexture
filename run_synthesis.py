import sys, struct, wave
import numpy as np
from scipy.io.wavfile import read

def open_wavefile(filename):
	print "READING: " + filename 
	try:
		[fs, wavefile] = read(filename, 'r')
	except:
		print "ERROR: could not read file"
		sys.exit(1)
	
	x = np.array(wavefile,dtype=float)
	dim = x.shape
	num_chan = 1
	if len(dim) > 1:
		num_chan = dim[1]
		for c in range(1, num_chan):
			x[:,c] = 1. * x[:,c] / max(abs(x[:,c])) 	# normalize
	else:
		x = 1. * x / max(abs(x)) 	# normalize
	num_frames = dim[0]

	print "\tsample rate: ", fs, "\n\t# samples: ", num_frames, "\n\t# channels: ", num_chan

	return x

if __name__ == "__main__":
	if len(sys.argv) > 1:
		filename = sys.argv[1]
		x = open_wavefile(filename)
	else:
		print "no user input"
		sys.exit(1)

	print x








