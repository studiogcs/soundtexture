# CS 229 - FALL 2015

# USAGE: 
#
# $ python run_synthesis.py <path to wave file>
#
import pickle

import sys, struct, wave
import collections
import numpy as np
import os
from scipy.io.wavfile import read
from scipy.signal import hilbert, resample
import sklearn.manifold
from wavio import readwav   #  https://github.com/WarrenWeckesser/wavio.git

from matplotlib import pyplot as plt  # for debugging

# Globals --------------------------------------------------------------------------
import traceback

default_options = {
	'rms': 0.01,  # desired root mean square
	'low_freq_limit': 20,  # Hz low end
	'num_audio_channels': 30,  # number of bands
	'env_fs': 400,  # sample rate of band envelope, for resampling
	'comp_exp': 0.3,  # compression exponent
	'num_mod_channels': 20,
	'low_mod_limit': 0.5,
	'Q': 2,
	'corr_env_intervals': np.array(
		[1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 18, 22, 28, 36, 45, 57, 73, 92, 116, 148, 187, 237, 301]),
	'low_c2_limit': 1,
}

# Classes ---------------------------------------------------------------------------
# Structure containing statistical measures; initialize with number of subbands
class Compute(object):
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

	def __init__(self, soundfile, fs, **options):
		self.soundfile = soundfile
		self.fs = fs
		self.N = len(soundfile)
		self.options = options
		# [setattr(self, key, value) for key,value in options.iteritems()]

		self.w_len = 1. * self.N / self.fs * self.options['env_fs']  # downsample length

		# make window
		num_windows = np.round(1. * self.N / fs) + 2
		onset_len = np.round(self.w_len / (num_windows - 1))
		w = make_cos_window(onset_len, self.w_len)
		self.w = w.reshape(self.w_len, 1)

		# make subbands using freq limit units: Hz -> erb
		print 'generating band filters...'
		low = self.options['low_freq_limit']
		high = 1. * self.fs / 2
		cutoffs = erb2freq(np.arange(
			freq2erb(low),
			freq2erb(high),
			(freq2erb(high) - freq2erb(low)) / (self.options['num_audio_channels'] + 1)
		))
		self.filters = make_erb_cos_filters(self.N, self.fs, self.options['num_audio_channels'], cutoffs)

		print 'computing subbands...'
		self.subbands = apply_filters(self.soundfile, self.filters)

		# extract envelopes for each band
		self.subband_env = np.abs(hilbert(self.subbands.T).T)

		# apply compression nonlinearity
		self.subband_comp = self.subband_env ** self.options['comp_exp']

		# resample
		self.subband_resampled = resample(self.subband_comp, self.w_len)
		self.subband_resampled[self.subband_resampled < 0] = 0

	def mod_filters(self):
		# make modulation filters
		print 'computing modulation filters...'
		num_mod_channels = self.options['num_mod_channels']
		mod_low = self.options['low_mod_limit']
		mod_high = 1. * self.options['env_fs'] / 2
		ds = (np.log2(mod_high) - np.log2(mod_low)) / (num_mod_channels - 1)
		self.mod_cutoffs = 2 ** ( np.arange(np.log2(mod_low), np.log2(mod_high) + ds, ds) )
		self.mod_filters = make_octave_cos_filters(
			self.w_len, self.options['env_fs'], num_mod_channels, self.mod_cutoffs, Q=self.options['Q'])

	def acf_filters(self):
		# make autocorrelation filters
		print 'computing autocorrelation filters...'
		self.intervals = self.options['corr_env_intervals']
		self.corr_filters = make_corr_filters(self.w_len, self.options['env_fs'], self.intervals)

	def c_filters(self):
		# make C2 filters
		print 'computing C1 and C2 filters'
		c2_low = self.options['low_c2_limit']
		c2_hi = 1. * self.options['env_fs']/ 2
		c2_cutoffs = c2_hi / (2 ** np.arange(0, 21))
		c2_cutoffs = np.flipud(c2_cutoffs[np.where(c2_cutoffs > c2_low)])
		self.c2_filters, self.c2_fc = make_octave_corr_filters(self.w_len, self.options['env_fs'], c2_cutoffs)

		# make C1 filters
		self.c1_filters = self.c2_filters[:, 1:]
		self.c1_fc = self.c2_fc[1:]

	def make_stats(self):
		self.mod_filters()
		self.acf_filters()
		self.c_filters()

		# compute statistics:
		print "Computing statistics"
		num_subbands = np.shape(self.subbands)[1]
		self.mean = np.mean(self.subbands, 0)
		self.var = np.var(self.subbands, 0)
		m0 = self.subbands - self.mean  # zero mean
		self.skew = np.mean(m0 ** 3, 0) / (np.mean(m0 ** 2, 0)) ** 1.5
		self.kurtosis = np.mean(m0 ** 4, 0) / (np.mean(m0 ** 2, 0)) ** 2

		self.e_mean = np.mean(self.subband_resampled * self.w, 0)
		env_m0 = self.subband_resampled - self.e_mean
		v2 = np.mean((env_m0 ** 2) * self.w, 0)
		self.e_var = np.sqrt(v2) / self.e_mean
		self.e_skew = np.mean((env_m0 ** 3) * self.w, 0) / (v2 ** 1.5)
		self.e_kurtosis = np.mean((env_m0 ** 4) * self.w, 0) / (v2 ** 2)

		self.e_auto_c = np.zeros((num_subbands, len(self.options['corr_env_intervals'])))
		self.mod_power = np.zeros((num_subbands, self.options['num_mod_channels']))
		self.mod_c2 = np.zeros((num_subbands, len(self.mod_cutoffs)))
		for i in range(0, num_subbands):
			self.e_auto_c[i, :] = envelope_autocorrelation(
				self.subband_resampled[:, i],
				self.corr_filters, self.options['corr_env_intervals'], self.w)
			self.mod_power[i, :] = modulation_power(self.subband_resampled[:, i], self.mod_filters, self. w)

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

	def features(self):
		return np.concatenate((
			self.mean,
			self.var,
			self.e_mean,
			self.e_var,
			self.e_auto_c[0],
			self.e_auto_c[1],

		))


def plots(self):
		# # DEBUG
		print 'plotting...'

		fig = plt.figure()
		fig.add_subplot(4, 1, 1)
		plt.plot(self.w)
		plt.ylabel('window')
		ax = fig.add_subplot(4, 1, 2)
		for i in range(0, self.options['num_audio_channels'] + 2):
			plt.plot(self.filters[:, i])
		plt.ylabel('filtered bands')
		ax.set_xscale('log')
		ax = fig.add_subplot(4, 1, 3)
		for i in range(0, self.options['num_audio_channels'] + 2):
			plt.plot(self.subbands[:, i])
		plt.ylabel('subbands')
		ax = fig.add_subplot(4, 1, 4)
		plt.plot(self.subband_env[:, 2])
		plt.ylabel('ex. envelope')
		plt.xlabel('samples')

		fig = plt.figure()
		fig.add_subplot(4, 1, 1)
		for i in range(0, self.options['num_mod_channels']):
			plt.plot(self.mod_filters[:, i])
		plt.ylabel('mod chans')
		fig.add_subplot(4, 1, 2)
		for i in range(0, len(self.intervals)):
			plt.plot(self.corr_filters[:, i])
		plt.ylabel('corr filters')
		fig.add_subplot(4, 1, 3)
		for i in range(0, np.shape(self.c2_filters)[1]):
			plt.plot(self.c2_filters[:, i])
		plt.ylabel('c2 filters')
		plt.show()

# Locals ----------------------------------------------------------------------------

# Loads a wav file from the complete path
def open_wavefile(filename, rms=.01):
	print "READING: " + filename
	try:
		[fs, wavefile] = read(filename)
		# [fs, width, wavefile] = readwav(filename)
	except:
		print traceback.format_exc()
		print "ERROR: could not read file"
		sys.exit(1)

	x = np.array(wavefile, dtype=float)
	dim = x.shape
	num_chan = 1

	# normalize
	if len(dim) > 1:
		num_chan = dim[1]
		for c in range(1, num_chan):
			rms = np.sqrt(np.mean(np.square(x[:, c])))
			x[:, c] = 1. * x[:, c] / rms * rms
	else:
		rms = rms = np.sqrt(np.mean(np.square(x)))
		x = 1. * x / rms * rms
	num_frames = dim[0]

	print "\tsample rate: ", fs, "\n\t# samples: ", num_frames, "\n\t# channels: ", num_chan

	return x, fs, num_frames


# Generates a raised cosine window
def make_cos_window(onset_len, w_len):
	w = np.ones(w_len)
	w[0:onset_len - 1] = 0.5 * (np.cos(np.arange(-onset_len + 1, 0) * np.pi / onset_len) + 1)
	w[w_len - onset_len:] = 0.5 * (np.cos(np.arange(0, onset_len) * np.pi / onset_len) + 1)

	return w


def freq2erb(f):
	return 9.265 * np.log(1 + f / (24.7 * 9.265))


def erb2freq(erb):
	return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)


# cosine filters based on ERB frequency band limits
def make_erb_cos_filters(N, fs, num_bands, cutoffs):
	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N - 1) / 2
		nyquist = 1. * fs / 2 * (1 - 1 / N)
	f = np.arange(0, N_f) * nyquist / N_f
	filters = np.zeros((N_f + 1, num_bands + 2))

	for i in range(0, num_bands):
		l = cutoffs[i]
		h = cutoffs[i + 2]
		l_ind = np.min(np.where(f > l))
		h_ind = np.max(np.where(f < h))
		avg = (freq2erb(l) + freq2erb(h)) / 2
		filters[l_ind:h_ind, i + 1] = np.cos(( freq2erb(f[l_ind:h_ind]) - avg ) / ( freq2erb(l) - freq2erb(h)) * np.pi)

	# add HPF as first low end and LPF for high end of the bandwidth
	l_ind = np.min(np.where(f > cutoffs[num_bands]))
	h_ind = np.max(np.where(f < cutoffs[1]))
	filters[0:h_ind, 0] = np.sqrt(1 - filters[0:h_ind, 1] ** 2)
	filters[l_ind:, num_bands + 1] = np.sqrt(1 - filters[l_ind:, num_bands] ** 2)

	return filters


# Apply multiple fft filtering to input vector x
def apply_filters(x, filters):
	N = np.shape(x)[0]
	filt_len, num_filters = np.shape(filters)
	X = np.fft.fft(x).repeat(num_filters).reshape(N, num_filters)

	if N % 2 == 0:
		fft_filters = np.concatenate((filters, np.flipud(filters[0:filt_len - 2, :])), axis=0)
	else:
		fft_filters = np.concatenate((filters, np.flipud(filters[0:filt_len - 1, :])), axis=0)

	fft_subbands = fft_filters * X

	return np.real(np.fft.ifft(fft_subbands.T).T)


# Apply vector fft filtering
def apply_filter(x, filter):
	N = len(x)
	filt_len = len(filter)
	X = np.fft.fft(x)

	if N % 2 == 0:
		fft_filter = np.concatenate((filter, np.flipud(filter[0:filt_len - 2])), axis=0)
	else:
		fft_filter = np.concatenate((filter, np.flipud(filter[0:filt_len - 1])), axis=0)

	return np.real(np.fft.ifft(fft_filter * X))


# Constant Q logarithmically spaced cosine filters
def make_octave_cos_filters(N, fs, num_bands, cutoffs, Q=2):
	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N - 1) / 2
		nyquist = 1. * fs / 2 * (1 - 1 / N)
	f = np.arange(0, N_f) * nyquist / N_f
	filters = np.zeros((N_f + 1, num_bands))

	for i in range(0, num_bands):
		bw = cutoffs[i] * 1. / Q
		l = cutoffs[i] - bw
		h = cutoffs[i] + bw
		l_ind = np.where(f > l)[0][0]
		h_ind = np.where(f < h)[0][-1]
		avg = cutoffs[i]
		filters[l_ind:h_ind, i] = np.cos(( f[l_ind:h_ind] - avg ) / (h - l) * np.pi)

	s = np.sum(filters ** 2, 1)
	filters = filters / np.sqrt(np.mean(s[(f >= cutoffs[3]) & (f <= cutoffs[-4])]))

	return filters


def make_corr_filters(N, fs, intervals):
	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N - 1) / 2
		nyquist = 1. * fs / 2 * (1 - 1 / N)
	f = np.arange(0, N_f) * nyquist / N_f

	num_bands = len(intervals)
	filters = np.zeros((N_f + 1, num_bands))

	for i in range(0, num_bands):
		if i == 0:
			h = 1. / (4. * intervals[i] / 1000)
			l = .5 / (4. * intervals[i] / 1000)
		else:
			h = 1. / (4. * (intervals[i] - intervals[i - 1]) / 1000)
			l = .5 / (4. * (intervals[i] - intervals[i - 1]) / 1000)

		if h > nyquist:
			filters[:, i] = np.ones(N_f + 1)
		else:
			l_ind = np.min(np.where(f > l))
			h_ind = np.max(np.where(f < h))
			if l_ind < h_ind:
				filters[0:l_ind - 1, i] = np.ones(l_ind - 1)
				filters[l_ind - 1:h_ind, i] = np.cos(
					( f[l_ind - 1:h_ind] - f[l_ind] ) / ( f[l_ind] - f[h_ind] ) * np.pi / 2)
			else:
				filters[0:l_ind - 1, i] = np.ones(l_ind - 1)

	return filters


def make_octave_corr_filters(N, fs, cutoffs):
	if N % 2 == 0:
		N_f = N / 2
		nyquist = 1. * fs / 2
	else:
		N_f = (N - 1) / 2
		nyquist = 1. * fs / 2 * (1 - 1 / N)
	f = np.arange(0, N_f) * nyquist / N_f

	center_freqs = cutoffs[0:-1]
	num_bands = len(center_freqs)

	filters = np.zeros((N_f + 1, num_bands))

	for i in range(0, num_bands):
		l = 1. * center_freqs[i] / 2
		h = cutoffs[i] * 2.
		l_ind = np.min(np.where(f > l))
		h_ind = np.max(np.where(f < h))
		avg = (np.log2(l) + np.log2(h)) / 2
		filters[l_ind:h_ind, i] = np.cos(( np.log2(f[l_ind:h_ind]) - avg ) / ( np.log2(h) - np.log2(l) ) * np.pi)

	return filters, center_freqs


# Shift vector by number of samples
def shift(v, lag):
	v = v.reshape(len(v), 1)
	if lag < 0:
		output = np.concatenate((v[-lag:], np.zeros((-lag, 1))), axis=0)
	else:
		output = np.concatenate((np.zeros((lag, 1)), v[0:-lag]), axis=0)
	return output


def envelope_autocorrelation(env, filters, intervals, w):
	w = w / np.sum(w)
	num_intervals = len(intervals)
	ac = np.zeros(num_intervals)
	for i in range(0, num_intervals):
		num_samples = intervals[i]
		f_env = apply_filter(env, filters[:, i])  # vector filter, vector input
		m = np.mean(f_env)
		f_env_m0 = f_env - m
		v = np.mean(f_env_m0 ** 2)
		ac[i] = np.sum(w * (shift(f_env_m0, -num_samples)) * shift(f_env_m0, num_samples)) / v
	return ac


def modulation_power(x, filters, w):
	num_bands = np.shape(filters)[1]
	N = len(x)
	x = x.reshape(N, 1)
	mod_x = apply_filters(x, filters)
	w = w / np.sum(w)
	v = np.sum(w * (x - np.sum(w * x)) ** 2)
	w_matrix = w.repeat(num_bands).reshape(N, num_bands)

	return np.sum(w_matrix * mod_x ** 2, 0) / v


def cache_to_disk(f):
	def wrapped(*args, **kwargs):
		redo = kwargs.pop('redo', False)
		key_args = [str(arg) for arg in args if isinstance(arg, collections.Hashable)]
		key = hash(f.func_name + '_' + '_'.join(key_args))
		if not os.path.exists('caches'):
			os.makedirs('caches')
		full_path = 'caches/%s.pkl' % key
		if redo:
			result = f(*args, **kwargs)
			pickle.dump(result, open(full_path, 'w+'))
		elif os.path.exists(full_path):
			result = pickle.load(open(full_path))
		else:
			result = f(*args, **kwargs)
			pickle.dump(result, open(full_path, 'w+'))
		return result
	return wrapped


@cache_to_disk
def get_features(filenames, limit):
	wins = []
	labels = []
	for filename in filenames:
		# extract features
		soundfile, fs, N = open_wavefile('wavefiles/' + filename, rms=default_options['rms'])
		if len(soundfile.shape) > 1:
			soundfile = soundfile.mean(1)
		soundfile = soundfile[::3]
		fs = fs // 3
		label = filename[:4].lower()
		stride = fs // 2
		win_size = fs
		n_wins = (N - win_size) // stride
		for i in xrange(min(limit, n_wins)):
			stats = Compute(soundfile[i * stride:i * stride + win_size], fs, **default_options)
			stats.make_stats()
			wins.append(stats.features())
			labels.append(label)
		# stats.display(0, None)
		# stats.plots()

	return np.array(wins), np.array(labels)

# Run ----------------------------------------------------------------------------------

if __name__ == "__main__":

	# if len(sys.argv) > 1:
	# 	filename = sys.argv[1]
	# 	soundfile, fs, N = open_wavefile(filename, rms=default_options['rms'])
	# else:
	# 	print "no user input"
	# 	sys.exit(1)

	filenames = (
		'Lecture Hall Chatter.wav',
		'Sunnyvale Station.wav',
		# 'Applause_-_enthusiastic2.wav',
		# 'Bubbling_water.wav',
		# 'Writing_with_pen_on_paper.wav',
		# 'white_noise_5s.wav',
		# 'chamber-choir-parallel-fifths.wav',

	)

	wins, labels = get_features(filenames, 150)
	print wins.shape
	m, s = np.mean(wins, 0), np.std(wins, 0)
	labels = np.array(labels)
	labels_set = [l[:4].lower() for l in filenames]

	train = np.random.rand(len(wins)) < .8

	m, v = np.mean(wins[train], 0), np.var(wins[train], 0)
	# scale = lambda X: (X - m) / s
	mod = sklearn.linear_model.LogisticRegression(C=.7).fit(wins[train], labels[train])
	# mod = sklearn.svm.SVC(kernel='rbf', C=.56).fit(wins[train], labels[train])
	print sklearn.metrics.confusion_matrix(labels[train], mod.predict(wins[train]))
	print sklearn.metrics.classification_report(labels[train], mod.predict(wins[train]))
	print sklearn.metrics.confusion_matrix(labels[~train], mod.predict(wins[~train]))
	print sklearn.metrics.classification_report(labels[~train], mod.predict(wins[~train]))

	X = sklearn.manifold.TSNE().fit_transform(wins)
	# V, S, U_t = np.linalg.svd(wins)
	# X = V[:, :2]

	colors = ['r', 'g', 'b', 'y', 'k']
	plt.figure()
	plt.scatter(X[:, 0], X[:, 1],
		c=[colors[labels_set.index(label)] for label in labels])
	# plt.show()
	print wins








