# CS 229 - FALL 2015

# USAGE: 
#
#		$ python run_synthesis.py <path to wave file>
#

import sys, struct, wave
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import hilbert, resample

from matplotlib import pyplot as plt 	# for debugging

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
g_low_c2_limit = 1

# Classes ---------------------------------------------------------------------------
# Structure containing statistical measures; initialize with number of subbands
class Stats:
	mean = np.array([])
	var = np.array([])
	skew = np.array([])
	kurtosis = np.array([])
	auto_c = np.array([])
	auto_c_power = np.array([])
	# envelope features
	e_mean = np.array([])	
	e_var = np.array([])
	e_skew = np.array([])
	e_kurtosis = np.array([])
	e_auto_c = np.array([])
	e_c = np.array([])
	# modulation band features
	mod_power = np.array([])
	mod_c1 = np.array([])
	mod_c2 = np.array([])

	def __init__(self,):
		return

	def display(self, start, end):
		print "\nmean:\n\t", self.mean[start:end]
		print "\nvar:\n\t", self.var[start:end]
		print "\nskew:\n\t", self.skew[start:end]
		print "\nkurtosis:\n\t", self.kurtosis[start:end]
		print "\nsubband autocorrelation: \n\t", self.auto_c[start:end]
		print "\nsubband autocorrelation power:\n\t", self.auto_c_power[start:end]
		print "\nenvelope mean:\n\t", self.e_mean[start:end]
		print "\nnenvelope var:\n\t", self.e_var[start:end]
		print "\nenvelope skew:\n\t", self.e_skew[start:end]
		print "\nenvelope kurtosis:\n\t", self.e_kurtosis[start:end]
		print "\nenvelope autocorrelation:\n\t", self.e_auto_c[start:end]
		print "\nenvelope correlation:\n\t", self.e_c[start:end]
		print "\nmodulation power:\n\t", self.mod_power[start:end]
		print "\nmodulation C1:\n\t", self.mod_c1[start:end]
		print "\nmodulation C2:\n\t", self.mod_c2[start:end]

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

	for i in range(0,num_bands):
		l = cutoffs[ i ]
		h = cutoffs[ i + 2 ]
		l_ind = np.min(np.where(f > l))
		h_ind = np.max(np.where(f < h))
		avg = (freq2erb(l) + freq2erb(h)) / 2
		filters[l_ind:h_ind, i + 1] = np.cos( ( freq2erb( f[ l_ind:h_ind ] ) - avg ) / ( freq2erb(l) - freq2erb(h)) * np.pi )

	# add HPF as first low end and LPF for high end of the bandwidth
	l_ind = np.min(np.where(f > cutoffs[num_bands]))
	h_ind = np.max(np.where(f < cutoffs[1]))
	filters[0:h_ind,0] = np.sqrt( 1 - filters[0:h_ind,1] ** 2 )
	filters[l_ind:, num_bands+1] = np.sqrt( 1 - filters[l_ind:, num_bands] ** 2 )
	
	return filters

# Apply multiple fft filtering to input vector x
def apply_filters(x, filters):
	N = np.shape(x)[0]
	filt_len, num_filters = np.shape(filters)
	X = np.fft.fft(x).repeat(num_filters).reshape(N, num_filters)

	if N % 2 == 0:
		fft_filters = np.concatenate( (filters, np.flipud(filters[0:filt_len-2,:])), axis=0)
	else:
		fft_filters = np.concatenate( (filters, np.flipud(filters[0:filt_len-1,:])), axis=0)

	fft_subbands = fft_filters * X

	return np.real(np.fft.ifft(fft_subbands.T).T)

# Apply vector fft filtering
def apply_filter(x, filter):
	N = len(x)
	filt_len = len(filter)
	X = np.fft.fft(x)

	if N % 2 == 0:
		fft_filter = np.concatenate( (filter, np.flipud(filter[0:filt_len-2])), axis=0)
	else:
		fft_filter = np.concatenate( (filter, np.flipud(filter[0:filt_len-1])), axis=0)

	return np.real(np.fft.ifft(fft_filter * X))

# Constant Q logarithmically spaced cosine filters
def make_octave_cos_filters(N, fs, num_bands, cutoffs):

	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N  - 1) / 2
		nyquist = 1. * fs/2 * (1-1/N)	
	f = np.arange(0,N_f) * nyquist  / N_f 
	filters = np.zeros( (N_f + 1, num_bands) )

	Q = g_Q
	for i in range(0, num_bands):
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

	for i in range(0,num_bands):
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
				filters[l_ind-1:h_ind, i] = np.cos( ( f[ l_ind-1:h_ind ] - f[ l_ind ] ) / ( f[l_ind] - f[ h_ind ] ) * np.pi/2 )
			else:
				filters[0:l_ind-1,i] = np.ones(l_ind-1)

	return filters

def make_octave_corr_filters(N, fs, cutoffs):

	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N  - 1) / 2
		nyquist = 1. * fs/2 * (1-1/N)	
	f = np.arange(0,N_f) * nyquist  / N_f 

	center_freqs = cutoffs[0:-1]
	num_bands = len(center_freqs)

	filters = np.zeros( (N_f+1, num_bands) )

	for i in range(0, num_bands):
		l = 1. * center_freqs[i] / 2
		h = cutoffs[i] * 2.
		l_ind = np.min(np.where(f > l))
		h_ind = np.max(np.where(f < h))
		avg = (np.log2(l) + np.log2(h)) / 2
		filters[l_ind:h_ind, i] = np.cos( ( np.log2( f[l_ind:h_ind] ) - avg ) / ( np.log2(h) - np.log2(l) ) * np.pi )

	return filters, center_freqs

# Shift vector by number of samples
def shift(v, lag):
	v = v.reshape(len(v),1)
	if lag < 0:
		output = np.concatenate( (v[-lag:], np.zeros((-lag,1))), axis=0)
	else:
		output = np.concatenate( (np.zeros((lag,1)), v[0:-lag]), axis=0)
	return output

def envelope_autocorrelation(env, filters, intervals, w):
	w = w / np.sum(w)
	num_intervals = len(intervals)
	ac = np.zeros(num_intervals)
	for i in range(0,num_intervals):
		num_samples = intervals[i]
		f_env = apply_filter(env, filters[:,i]) 	# vector filter, vector input
		m = np.mean(f_env)
		f_env_m0 = f_env - m
		v = np.mean(f_env_m0**2)
		ac[i] = np.sum( w*(shift(f_env_m0,-num_samples))*shift(f_env_m0,num_samples) ) / v
	return ac

def modulation_power(x, filters, w):
	num_bands = np.shape(filters)[1]
	N = len(x)
	x = x.reshape(N,1)
	mod_x = apply_filters(x, filters)
	w = w/np.sum(w)
	v = np.sum(w*(x - np.sum(w*x))**2)
	w_matrix = w.repeat(num_bands).reshape(N,num_bands)

	return np.sum( w_matrix * mod_x**2, 0) / v

def compute_statistics(soundfile, fs, N):
	N = len(soundfile)
	num_audio_channels = g_num_audio_channels

	env_fs = g_env_fs
	w_len = 1. * N / fs * env_fs 	# downsample length

	# make subbands using freq limit units: Hz -> erb
	print 'generating band filters...'
	low = g_low_freq_limit
	high = 1. * fs / 2
	cutoffs = erb2freq( np.arange(freq2erb(low), freq2erb(high), (freq2erb(high) - freq2erb(low)) / (num_audio_channels+1)) )
	filters = make_erb_cos_filters(N, fs, num_audio_channels, cutoffs)

	print 'computing subbands...'
	subbands = apply_filters(soundfile, filters)
	
	# extract envelopes for each band
	subband_env = np.abs(hilbert(subbands.T).T)

	# apply compression nonlinearity
	subband_comp = subband_env**g_comp_exp

	# resample
	subband_resampled = resample(subband_comp, w_len)
	subband_resampled[subband_resampled < 0] = 0

	# make modulation filters
	print 'computing modulation filters...'
	num_mod_channels = g_num_mod_channels
	mod_low = g_low_mod_limit
	mod_high = 1. * env_fs / 2
	ds = (np.log2(mod_high) - np.log2(mod_low)) / (num_mod_channels - 1) 
	mod_cutoffs = 2**( np.arange( np.log2(mod_low),  np.log2(mod_high) + ds, ds) )
	mod_filters = make_octave_cos_filters(w_len, env_fs, num_mod_channels, mod_cutoffs)

	# make autocorrelation filters
	print 'computing autocorrelation filters...'
	intervals = g_corr_env_intervals
	corr_filters = make_corr_filters(w_len, env_fs, intervals)

	# make C2 filters
	print 'computing C1 and C2 filters'
	c2_low = g_low_c2_limit
	c2_hi = 1. * env_fs / 2
	c2_cutoffs = c2_hi / (2**np.arange(0,21))
	c2_cutoffs = np.flipud( c2_cutoffs[ np.where(c2_cutoffs > c2_low) ] )
	c2_filters, c2_fc = make_octave_corr_filters(w_len, env_fs, c2_cutoffs)

	# make C1 filters
	c1_filters = c2_filters[:,1:]
	c1_fc = c2_fc[1:]

	# make window
	num_windows = np.round(1. * N / fs) + 2
	onset_len = np.round(w_len/ (num_windows-1))
	w = make_cos_window(onset_len, w_len)
	w = w.reshape(w_len,1)

	# compute statistics:
	print "Computing statistics"
	num_subbands = np.shape(subbands)[1]
	stats = Stats()
	stats.mean = np.mean(subbands,0)
	stats.var = np.var(subbands,0)
	m0 = subbands - stats.mean 	# zero mean
	stats.skew = np.mean(m0**3,0) / (np.mean(m0**2,0))**1.5
	stats.kurtosis = np.mean(m0**4,0) / (np.mean(m0**2,0))**2

	stats.e_mean = np.mean(subband_resampled * w,0)
	env_m0 = subband_resampled - stats.e_mean
	v2 = np.mean((env_m0**2)*w,0)
	stats.e_var = np.sqrt(v2) / stats.e_mean
	stats.e_skew = np.mean((env_m0**3)*w,0) / (v2**1.5)
	stats.e_kurtosis = np.mean((env_m0**4)*w,0) / (v2**2)

	stats.e_auto_c = np.zeros((num_subbands, len(intervals)))
	stats.mod_power = np.zeros((num_subbands, num_mod_channels))
	stats.mod_c2 = np.zeros((num_subbands, len(mod_cutoffs)))
	for i in range(0,num_subbands):
		stats.e_auto_c[i,:] = envelope_autocorrelation(subband_resampled[:,i], corr_filters, intervals, w)
		stats.mod_power[i,:] = modulation_power(subband_resampled[:,i], mod_filters, w)

	stats.display(0,2)
	exit(1)

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
	ax.set_xscale('log')
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
	fig.add_subplot(4,1,3)
	for i in range(0, np.shape(c2_filters)[1]):
		plt.plot(c2_filters[:,i])
	plt.ylabel('c2 filters')
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








