# CS 229 - FALL 2015

# USAGE: 
#
#		$ python run_synthesis.py <path to wave file>
#

import sys, struct, wave
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import hilbert

# DEBUG
from matplotlib import pyplot as plt

# Globals --------------------------------------------------------------------------
g_rms = 0.01 # desired root mean square
g_low_freq_limit = 20
g_num_audio_channels = 30
g_env_fs = 400
# Locals ----------------------------------------------------------------------------

# Loads a wav file from the complete path
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

	# normalize
	if len(dim) > 1:
		num_chan = dim[1]
		for c in range(1, num_chan):
			rms = np.sqrt(np.mean(np.square(x[:,c])))
			x[:,c] = 1. * x[:,c] / rms * g_rms 
	else:
		rms = rms = np.sqrt(np.mean(np.square(x)))
		x = 1. * x / rms * g_rms
	num_frames = dim[0]

	print "\tsample rate: ", fs, "\n\t# samples: ", num_frames, "\n\t# channels: ", num_chan

	return x, fs, num_frames

# Generates a raised cosine window
def make_cos_window(onset_len, w_len):

	w = np.ones(w_len)
	w[0:onset_len-1] = 0.5 * (np.cos(np.arange(-onset_len+1,0) * np.pi / onset_len)+1)
	w[w_len - onset_len:] = 0.5 * (np.cos(np.arange(0,onset_len) * np.pi / onset_len) + 1)

	return w

def freq2erb(f):
	return 9.265*np.log(1+f / (24.7*9.265))

def erb2freq(erb):
	return 24.7 * 9.265  * (np.exp(erb/9.265) - 1)

# Erb cosine filters
def make_erb_cos_filters(N, fs, num_audio_channels):

	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N  - 1) / 2
		nyquist = 1. * fs/2 * (1-1/N)	
	f = np.arange(0,N_f) * nyquist / N_f 
	filters = np.zeros( (N_f + 1, num_audio_channels + 2) )

	# freq limits (Hz -> erb)
	low = g_low_freq_limit
	high = nyquist
	cutoffs = erb2freq( np.arange(freq2erb(low), freq2erb(high), (freq2erb(high) - freq2erb(low)) / (num_audio_channels+1)) )

	for i in range(0,num_audio_channels-1):
		l = cutoffs[ i ]
		h = cutoffs[ i + 2 ]
		l_ind = np.min(np.where(f > l))
		h_ind = np.max(np.where(f < h))
		avg = (freq2erb(l) + freq2erb(h)) / 2
		filters[l_ind:h_ind, i + 1] = np.cos( ( freq2erb( f[ l_ind:h_ind ] ) - avg ) / ( freq2erb(l) - freq2erb(h)) * np.pi )

	# add HPF as first low end and LPF for high end of the bandwidth
	l_ind = np.min(np.where(f > cutoffs[num_audio_channels-1]))
	h_ind = np.max(np.where(f < cutoffs[1]))
	filters[0:h_ind,0] = np.sqrt( 1 - filters[0:h_ind,1] ** 2 )
	filters[l_ind:, num_audio_channels+1] = np.sqrt( 1 - filters[l_ind:,num_audio_channels-1] ** 2 )
	
	return filters, cutoffs

def make_subbands(x, filters):
	N = np.shape(x)[0]
	filt_len, num_filters = np.shape(filters)
	X = np.fft.fft(x).repeat(num_filters).reshape(N, num_filters)

	if N % 2 == 0:
		fft_filters = np.concatenate( (filters, np.flipud(filters[0:filt_len-2,:])), axis=0)
	else:
		fft_filters = np.concatenate( (filters, np.flipud(filters[0:filt_len-1,:])), axis=0)

	fft_subbands = fft_filters * X

	## DEBUG
	# print x[0:4]
	# print X[0:4,0]
	# print fft_filters[0:4,0]
	# print fft_subbands[0:4,0]
	# fig = plt.figure()
	# plt.plot(fft_filters[:,20]) # filter
	# plt.plot(np.abs(X)) # FFT
	# plt.plot(np.abs(fft_subbands[:,20]))
	# plt.show()
	##

	return np.real(np.fft.ifft(fft_subbands.T).T)

def compute_statistics(soundfile, fs, N):
	stats = []
	N = len(soundfile)
	num_audio_channels = g_num_audio_channels

	# 1 second window
	env_fs = g_env_fs
	total_len = 1. * N / fs * env_fs
	num_windows = np.round(1. * N / fs) + 2
	onset_len = np.round(total_len/ (num_windows-1))
	w = make_cos_window(onset_len, total_len)

	# generate erb cosine filters
	print 'generating band filters...'
	filters, cutoffs = make_erb_cos_filters(N, fs, num_audio_channels)

	# make subbands
	print 'computing subbands...'
	subbands = make_subbands(soundfile, filters)

	# take envelopes
	subband_env = np.abs(hilbert(subbands.T).T)

	## DEBUG
	print 'plotting...'
	fig = plt.figure()
	fig.add_subplot(3,1,1)
	plt.plot(w)
	plt.ylabel('window')
	ax = fig.add_subplot(3,1,2)
	for i in range(0,num_audio_channels + 2):
		plt.plot(filters[:,i])
	plt.ylabel('filtered bands')
	ax = fig.add_subplot(3,1,3)
	# for i in range(0,num_audio_channels + 2):
	# 	plt.plot(subbands[:,i])
	# plt.ylabel('subbands')
	plt.plot(subband_env[:,2])
	plt.ylabel('band envelopes')
	plt.xlabel('samples')
	plt.show()
	##

	return stats

# Run ----------------------------------------------------------------------------------

if __name__ == "__main__":
	if len(sys.argv) > 1:
		filename = sys.argv[1]
		soundfile, fs, N = open_wavefile(filename)
	else:
		print "no user input"
		sys.exit(1)

	# extract features
	stats = compute_statistics(soundfile, fs, N)








