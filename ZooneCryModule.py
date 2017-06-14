# coding: utf-8

###############################
# ZooneCryModule : 動物の鳴き声認識共通スクリプト
# developed by Keisuke Yoshida
###############################

# import library
import wave
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.fftpack.realtransforms

class ZooneCryModule: 
	name = "abc"
	dataLength = 10
	dataDirectry = "./data/wav/"
	classNameArray = ["dog", "cat"]


	def __init__(self, name):
		self.name = name

	def getName(self):
		return self.name

	def getClassNameArray(self):
		return self.classNameArray

	def getDataLength(self):
		return self.dataLength


class MFCCModule:
	p = 0.97 # for human
	numChannels = 20 # channel
	nceps = 12 # number of dimension
	cuttime = 0.8 # sample sound time
	nfft = 2048 # 1024, 2048, 4096

	def __init__(self):
		print("MFCCModule Constructor")

	def readWavFile(self, filename):
		wf = wave.open(filename, "r")
		fs = wf.getframerate()
		x = wf.readframes(wf.getnframes())
		x = np.frombuffer(x, dtype="int16") / 32768.0 # (-1, 1) normalization
		wf.close()
		return x, float(fs)

	def convertHz2Mel(self, f):
		"""convert Hz to mel"""
		return 1127.01048 * np.log(f / 700.0 + 1.0)

	def convertMel2Hz(self, m):
		"""convert mel to Hz """
		return 700.0 * (np.exp(m / 1127.01048) - 1.0)

	def createMelFilterBank(self, fs, nfft, numChannels):
		""" create mel filter bank """
		
		# Nyquist frequency
		fmax = fs / 2
		melmax = self.convertHz2Mel(fmax)

		# max count of frequency index
		nmax = nfft / 2

		# resolution of frequency
		df = fs / nfft

		# center of frequency on mel
		dmel = melmax / (numChannels + 1)
		melcenters = np.arange(1, numChannels + 1) * dmel

		# center of frequency on hz
		fcenters = self.convertMel2Hz(melcenters)

		# convert fcenters to frequency index
		indexcenter = np.round(fcenters / df)

		# index of entry point on each filters 
		indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))

		# index of end point on each filters
		indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

		filterbank = np.zeros((numChannels, int(nmax)))

		for c in np.arange(0, numChannels):
			increment = 1.0 / (indexcenter[c] - indexstart[c])
			
			for i in np.arange(indexstart[c], indexcenter[c]):
				i = int(i)
				filterbank[c, i] = (i - indexstart[c]) * increment

			decrement = 1.0 / (indexstop[c] - indexcenter[c])

			for i in np.arange(indexcenter[c], indexstop[c]):
				i = int(i)
				filterbank[c,i] = 1.0 - ((i - indexcenter[c]) * decrement)

		return filterbank, fcenters

	def preEmphasis(self, signal, p):
		""" pre enmphasis filter """
		# FIR filter (1.0 , -p)
		return scipy.signal.lfilter([1.9, -p], 1, signal)

	def mfcc(self, signal, nfft, fs, nceps):
		""" parameter of MFCC """
		# signal : 
		# nfft : sample count of fft
		# nceps : demension of MFCC
	
		# 1. pre emphasis filter
		signal = self.preEmphasis(signal, self.p)

		# 2. hamming window
		hammingWindow = np.hamming(len(signal))
		signal = signal * hammingWindow

		# 3. amplitude spectrum
		spec = np.abs(np.fft.fft(signal, nfft))[:int(nfft/2)]
		fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:int(nfft/2)]

		# 4. make mel filter bank
		df = fs / nfft
		filterbank, fcenters = self.createMelFilterBank(fs, nfft, self.numChannels)

		# 5. sum of amplitude spectrum filtered with mel filter bank
		mspec = np.log10(np.dot(spec, filterbank.T))

		# 6. Discrete cosine transform
		ceps = scipy.fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)

		# return coefficient of less dimension element
		return ceps[:nceps]

	def getFeature(self,wavfile):
		# load sound file
		wav, fs = self.readWavFile(wavfile)
		t = np.arange(0.0, len(wav) / fs, 1/fs)

		# get wave around center
		center = len(wav) / 2
		wavdata = wav[int(center - self.cuttime/2*fs) : int(center + self.cuttime/2*fs)]

		ceps = self.mfcc(wavdata, self.nfft, fs, self.nceps)
		return ceps.tolist()

	def getP(self):
		return self.p


# test methods
# if __name__ == '__main__':
# 	test = ZooneCryModule("test")
# 	print(test.getName())
# 	print(test.getClassNameArray())
# 	print(test.getDataLength())
# 	

if __name__ == '__main__':
	mfcm = MFCCModule()
	wavfile = "./data/wav/cat/cat2.wav"
	print(mfcm.getP())
	features = mfcm.getFeature(wavfile)
	print(features)
    