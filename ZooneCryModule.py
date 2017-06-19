# coding: utf-8

###############################
# ZooneCryModule : 動物の鳴き声認識共通スクリプト
# developed by Keisuke Yoshida
###############################
import os

# import library for mfcc
import wave
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.fftpack.realtransforms

# import library for support vector machine
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib
import sys
import glob
import csv
import random
import itertools

class ZooneCryModule: 
	name = "zoone cry"
	dataLength = 4
	
	absPathName = os.path.dirname(os.path.abspath(__file__))

	soundDataDirectry =os.path.normpath(os.path.join(absPathName,"./data/wav/")) #"./data/wav/"#
	predictDataDirectry = os.path.normpath(os.path.join(absPathName,"./data/predict/")) #"./data/predict/"#

	featureTrainDataFileName = "/feature_train_data.txt"
	featureTestDataFileName = "/feature_test_data.txt"

	trainDataFileName = "/train_data.txt"
	trainLabelFileName = "/train_label.txt"

	testDataFileName = "/test_data.txt"
	testLabelFileName = "/test_label.txt"
	
	labels = ["dog", "cat","noise"]
	noise_nums = list(range(1,5)) # 1,2,3,4
	level_nums = list(range(0,10)) #0,1,2,3,4,5,6,7,8,9

	train_data = np.empty((0, 12),float)
	train_label = np.array([])

	test_data = np.empty((0, 12), float)
	test_label = np.array([])

	feature_train_data = np.empty((0,13))
	feature_test_data = np.empty((0,13))

	def __init__(self, name):
		#print("ZooneCryModule Constructor")
		self.name = name
		self.MFCCModule = MFCCModule()
		self.SVMModule = SVMModule()

	def getName(self):
		return self.name

	def getLabels(self):
		return self.labels

	def getDataLength(self):
		return self.dataLength

	def createTrainCryData(self):
		# import label1_1.wav, label1_2.wav, ...
		# TODO level parameter
		random.shuffle(self.noise_nums)
		random.shuffle(self.labels)
		for label in self.labels:
			for noise_num in self.noise_nums:
				files_name = glob.glob(self.soundDataDirectry+"%s/%s_%d.wav" % (label,label,noise_num))
				#print(files_name)
				for file_name in files_name:
					feature = self.MFCCModule.getFeature(file_name)
					if len(self.train_data) == 0:
						self.train_data=feature
					else:
						self.train_data=np.vstack((self.train_data,feature))
					self.train_label=np.append(self.train_label,label)
		feature_train_data = np.hstack((self.train_label.reshape(len(self.train_label),1),self.train_data))
		#print(feature_train_data)
		with open(self.predictDataDirectry+self.featureTrainDataFileName, "w") as f:
			writer = csv.writer(f)
			writer.writerows(feature_train_data)
		with open(self.predictDataDirectry+self.trainDataFileName, "w") as f:
			writer = csv.writer(f)
			writer.writerows(self.train_data)
		with open(self.predictDataDirectry+self.trainLabelFileName, "w") as f:
			writer = csv.writer(f)
			writer.writerow(self.train_label)
		self.feature_train_data = feature_train_data

	def createTestCryData(self):
		random.shuffle(self.noise_nums)
		random.shuffle(self.labels)
		for label in self.labels:
			noise_num = self.noise_nums[0]
			files_name = glob.glob(self.soundDataDirectry+"%s/%s_%d.wav" % (label,label,noise_num))
			#print(files_name)
			for file_name in files_name:
				feature = self.MFCCModule.getFeature(file_name)
				if len(self.test_data) == 0:
					self.test_data=feature
				else:
					self.test_data=np.vstack((self.test_data,feature))
				self.test_label=np.append(self.test_label,label)
		feature_test_data = np.hstack((self.test_label.reshape(len(self.test_label),1),self.test_data))
		#print(feature_test_data)
		with open(self.predictDataDirectry+self.featureTestDataFileName, "w") as f:
			writer = csv.writer(f)
			writer.writerows(feature_test_data)
		with open(self.predictDataDirectry+self.testDataFileName, "w") as f:
			writer = csv.writer(f)
			writer.writerows(self.test_data)
		with open(self.predictDataDirectry+self.testLabelFileName, "w") as f:
			writer = csv.writer(f)
			writer.writerow(self.test_label)
		self.feature_test_data = feature_test_data

	def readFeatureTrainCryData(self):
		with open(self.predictDataDirectry+self.featureTrainDataFileName, "r") as f:
			reader = csv.reader(f)
			feature_train_data = np.empty((0,self.MFCCModule.nceps+1))
			# header = next(reader)
			for row in reader:
				if len(feature_train_data) == 0:
					feature_train_data = row
				else:
					feature_train_data = np.vstack((feature_train_data, row))
			self.feature_train_data = feature_train_data

	def readFeatureTestCryData(self):
		with open(self.predictDataDirectry+self.featureTestDataFileName, "r") as f:
			reader = csv.reader(f)
			feature_test_data = np.empty((0,self.MFCCModule.nceps+1))
			for row in reader:
				if len(feature_test_data) == 0:
					feature_test_data = row
				else:
					feature_test_data = np.vstack((feature_test_data, row))
			self.feature_test_data = feature_test_data

	def readTrainCryData(self):
		with open(self.predictDataDirectry+self.trainDataFileName, "r") as f:
			reader = csv.reader(f)
			train_data = np.empty((0,self.MFCCModule.nceps))
			for row in reader:
				if len(train_data) == 0:
					train_data = row
				else:
					train_data = np.vstack((train_data, row))
			self.train_data = train_data
		with open(self.predictDataDirectry+self.trainLabelFileName, "r") as f:
			reader = csv.reader(f)
			label_data = next(reader)
			self.label_data = label_data

	def readTestCryData(self):
		with open(self.predictDataDirectry+self.testDataFileName, "r") as f:
			reader = csv.reader(f)
			test_data = np.empty((0,len(self.labels)))
			for row in reader:
				if len(test_data) == 0:
					test_data = row
				else:
					test_data = np.vstack((test_data, row))
			self.test_data = test_data

	def readTestDataWithFileName(self, filename):
		feature = self.MFCCModule.getFeature(filename)
		return feature

	def trainSVMModel(self):
		self.SVMModule.trainSVM(self.train_data, self.label_data)

	def predictClass(self):
		self.SVMModule.predictClass(self.test_data[0])

	def predictClassWithExistData(self, filename):
		# if user input music file name
		# this system predict class with svm
		feature = self.readTestDataWithFileName(filename)
		self.SVMModule.predictClassWithExistData(feature)

class MFCCModule:
	p = 0.97 # for human
	numChannels = 20 # channel
	nceps = 12 # number of dimension
	cuttime = 0.8 # sample sound time
	nfft = 2048 # 1024, 2048, 4096


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

# read/write train file and predict class
class SVMModule:
	absPathName = os.path.dirname(os.path.abspath(__file__))

	predictDataDirectry = os.path.normpath(os.path.join(absPathName,"./data/predict/"))
	predictDataFileName = "/predict.pkl"
	trainDataFileName = "/train_data.txt"
	testDataFileName = "/test_data.txt"

	def __init__(self):
		self.clf = svm.SVC()
		#print("constructorn on SVMModule")
		
	def readPredictModel(self, filename):
		return ""

	def writePredictModel(self, filename):

		return ""

	def readTrainDataFile(self, filename):
		return ""

	def writeTrainDataFile(self, trainData, labelData):
		joinData = np.hstack((labelData.reshape(len(labelData),1),trainData))
		with open(self.predictDataDirectry+self.featureTrainDataFileName,"w") as f:
			writer=csv.writer(f)
			writer.writerows(joinData)

	def trainSVM(self, trainData, labelData):
		self.clf.fit(trainData, labelData)
		joblib.dump(self.clf, self.predictDataDirectry+self.predictDataFileName)

	def predictClass(self, testData):
		result = self.clf.predict(testData.reshape(1,-1))
		print(result[0])
		#sys.stdout.flush()

	def predictClassWithExistData(self, testData): 
		self.clf = joblib.load(self.predictDataDirectry+self.predictDataFileName) 
		result = self.clf.predict(np.asarray(testData).reshape(1,-1))
		print(result[0])
		#sys.stdout.flush()

#test methods
# if __name__ == '__main__':
# 	a = range(1,5)
# 	for i in a:
# 		print(i)

# if __name__ == '__main__':
# 	mfcm = MFCCModule()
# 	wavfile = "./data/wav/cat/cat2.wav"
# 	print(mfcm.getP())
# 	features = mfcm.getFeature(wavfile)
# 	print(features)

#if __name__ == '__main__':
	#zcm = ZooneCryModule("abc")
	# zcm.createTrainCryData()
	# zcm.createTestCryData()
	# zcm.readTrainCryData()
	# zcm.readTestCryData()
	# zcm.trainSVMModel()
	# zcm.predictClassWithExistData('./data/wav/noise/noise_3.wav')

