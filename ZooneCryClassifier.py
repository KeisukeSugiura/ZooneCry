###############################
# ZooneCryClassifier : 動物の鳴き声認識分類スクリプト
# developed by Keisuke Yoshida
###############################
import os

# import library
from ZooneCryModule import ZooneCryModule
from ZooneCryModule import MFCCModule
from ZooneCryModule import SVMModule


#if __name__ == '__main__':
zcm = ZooneCryModule("abc")
absPathName = os.path.dirname(os.path.abspath(__file__))

	#zcm.createTrainCryData()
	#zcm.createTestCryData()
	#zcm.readTrainCryData()
	#zcm.readTestCryData()
	#zcm.trainSVMModel()
zcm.predictClassWithExistData(os.path.normpath(os.path.join('./modules/ZooneCry/data/wav/test.wav')))
# else:
# 	zcm = ZooneCryModule("check")
# 	zcm.predictClass('./data/wav/test.wav')