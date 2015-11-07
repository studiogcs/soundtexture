# CS 229 - FALL 2015

# USAGE: 
#
#		$ python run_synthesis.py <path to wave file>
#

import sys, struct, wave
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import hilbert, resample

# DEBUG
from matplotlib import pyplot as plt

# Globals --------------------------------------------------------------------------
g_rms = 0.01					# desired root mean square
g_low_freq_limit = 20 		# Hz low end
g_num_audio_channels = 30 	# number of bands
g_env_fs = 400 		# sample rate of band envelope, for resampling
g_comp_exp = 0.3	# compression exponent
g_num_mod_channels = 20
g_low_mod_limit = 0.5
g_Q = 2
g_corr_env_intervals = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 18, 22, 28, 36, 45, 57, 73, 92, 116, 148, 187, 237, 301])
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

# cosine filters based on ERB frequency band limits
def make_erb_cos_filters(N, fs, num_bands, cutoffs):

	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N  - 1) / 2
		nyquist = 1. * fs/2 * (1-1/N)	
	f = np.arange(0,N_f) * nyquist / N_f 
	filters = np.zeros( (N_f + 1, num_bands + 2) )

	for i in range(0,num_bands-1):
		l = cutoffs[ i ]
		h = cutoffs[ i + 2 ]
		l_ind = np.min(np.where(f > l))
		h_ind = np.max(np.where(f < h))
		avg = (freq2erb(l) + freq2erb(h)) / 2
		filters[l_ind:h_ind, i + 1] = np.cos( ( freq2erb( f[ l_ind:h_ind ] ) - avg ) / ( freq2erb(l) - freq2erb(h)) * np.pi )

	# add HPF as first low end and LPF for high end of the bandwidth
	l_ind = np.min(np.where(f > cutoffs[num_bands-1]))
	h_ind = np.max(np.where(f < cutoffs[1]))
	filters[0:h_ind,0] = np.sqrt( 1 - filters[0:h_ind,1] ** 2 )
	filters[l_ind:, num_bands+1] = np.sqrt( 1 - filters[l_ind:,num_bands-1] ** 2 )
	
	return filters

def make_subbands(x, filters):
	N = np.shape(x)[0]
	filt_len, num_filters = np.shape(filters)
	X = np.fft.fft(x).repeat(num_filters).reshape(N, num_filters)

	if N % 2 == 0:
		fft_filters = np.concatenate( (filters, np.flipud(filters[0:filt_len-2,:])), axis=0)
	else:
		fft_filters = np.concatenate( (filters, np.flipud(filters[0:filt_len-1,:])), axis=0)

	fft_subbands = fft_filters * X

	return np.real(np.fft.ifft(fft_subbands.T).T)

# constant Q logarithmically spaced cosine filters
def make_log2_cos_filters(N, fs, num_bands, cutoffs):

	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N  - 1) / 2
		nyquist = 1. * fs/2 * (1-1/N)	
	f = np.arange(0,N_f) * nyquist  / N_f 
	filters = np.zeros( (N_f + 1, num_bands) )

	Q = g_Q
	for i in range(0, num_bands-1):
		bw = cutoffs[ i ] * 1. / Q
		l = cutoffs[ i ] - bw
		h = cutoffs[ i ] + bw
		l_ind = np.where(f > l)[0][0]
		h_ind = np.where(f < h)[0][-1]
		avg = cutoffs[ i ]
		filters[l_ind:h_ind, i] = np.cos( ( f[ l_ind:h_ind ] - avg ) / (h - l) * np.pi )

	s =  np.sum(filters ** 2,1)
	filters = filters / np.sqrt( np.mean( s[ (f >= cutoffs[3]) & (f <= cutoffs[-4])  ] ) )

	return filters

def make_corr_filters(N, fs, intervals):

	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N  - 1) / 2
		nyquist = 1. * fs/2 * (1-1/N)	
	f = np.arange(0,N_f) * nyquist  / N_f 

	num_bands = len(intervals)
	filters = np.zeros( (N_f + 1, num_bands) )

	for i in range(0,num_bands-1):
		if i == 0:
			h = 1./ (4.*intervals[i] / 1000)
			l = .5 / (4.*intervals[i]/1000)
		else:
			h = 1. / (4.*(intervals[i] - intervals[i-1]) / 1000)
			l = .5 / (4.*(intervals[i] - intervals[i-1]) / 1000)

		if h > nyquist:
			filters[:,i] = np.ones(N_f + 1)
		else:
			l_ind = np.min(np.where(f > l))
			h_ind = np.max(np.where(f < h))
			if l_ind < h_ind:
				filters[0:l_ind-1, i] = np.ones(l_ind-1)
				filters[l_ind:h_ind, i] = np.cos( ( f[ l_ind:h_ind ] - f[ l_ind ] ) / ( f[l_ind] - f[ h_ind ] ) * np.pi/2 )
			else:
				filters[0:l_ind-1,i] = np.ones(l_ind-1)

	return filters

def compute_statistics(soundfile, fs, N):
	stats = []
	N = len(soundfile)
	num_audio_channels = g_num_audio_channels

	env_fs = g_env_fs
	dsamp_len = 1. * N / fs * env_fs 	# downsample length

	# make subbands using freq limit units: Hz -> erb
	print 'generating band filters...'
	low = g_low_freq_limit
	high = 1. * fs / 2
	cutoffs = erb2freq( np.arange(freq2erb(low), freq2erb(high), (freq2erb(high) - freq2erb(low)) / (num_audio_channels+1)) )
	filters = make_erb_cos_filters(N, fs, num_audio_channels, cutoffs)

	print 'computing subbands...'
	subbands = make_subbands(soundfile, filters)
	
	# extract envelopes for each band
	subband_env = np.abs(hilbert(subbands.T).T)

	# apply compression nonlinearity
	subband_comp = subband_env**g_comp_exp

	# resample
	subband_resampled = resample(subband_comp, dsamp_len)
	subband_resampled[subband_resampled < 0] = 0

	# make modulation filters
	num_mod_channels = g_num_mod_channels
	mod_len = dsamp_len
	mod_low = g_low_mod_limit
	mod_high = 1. * env_fs / 2
	ds = (np.log2(mod_high) - np.log2(mod_low)) / (num_mod_channels - 1) 
	mod_cutoffs = 2**( np.arange( np.log2(mod_low),  np.log2(mod_high) + ds, ds) )
	mod_filters = make_log2_cos_filters(mod_len, env_fs, num_mod_channels, mod_cutoffs)

	# make autocorrelation filters
	intervals = g_corr_env_intervals
	corr_filters = make_corr_filters(mod_len, env_fs, intervals)

	# make window
	num_windows = np.round(1. * N / fs) + 2
	onset_len = np.round(dsamp_len/ (num_windows-1))
	w = make_cos_window(onset_len, dsamp_len)

	## DEBUG
	print 'plotting...'
	fig = plt.figure()
	fig.add_subplot(4,1,1)
	plt.plot(w)
	plt.ylabel('window')
	ax = fig.add_subplot(4,1,2)
	for i in range(0,num_audio_channels + 2):
		plt.plot(filters[:,i])
	plt.ylabel('filtered bands')
	ax = fig.add_subplot(4,1,3)
	for i in range(0, num_audio_channels + 2):
		plt.plot(subbands[:,i])
	plt.ylabel('subbands')
	ax = fig.add_subplot(4,1,4)
	plt.plot(subband_env[:,2])
	plt.ylabel('ex. envelope')
	plt.xlabel('samples')

	fig = plt.figure()
	fig.add_subplot(4,1,1)
	for i in range(0, num_mod_channels):
		plt.plot(mod_filters[:,i])
	plt.ylabel('mod chans')
	fig.add_subplot(4,1,2)
	for i in range(0, len(intervals)):
		plt.plot(corr_filters[:,i])
	plt.ylabel('corr filters')

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








