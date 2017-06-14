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




# test methods
# if __name__ == '__main__':
# 	test = ZooneCryModule("test")
# 	print(test.getName())
# 	print(test.getClassNameArray())
# 	print(test.getDataLength())
