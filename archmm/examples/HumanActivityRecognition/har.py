# -*- coding: utf-8 -*-

import os, sys
from random import shuffle as random_shuffle
import numpy as np
from scipy.io import loadmat

import theano

from archmm.core import *

BASE_PATH = "C://Users/Xanto183/Downloads/USC-HAD/USC-HAD"

def loadDS():
	dataset = list()
	for filename in os.listdir(BASE_PATH):
		folder_name = os.path.join(BASE_PATH, filename)
		if os.path.isdir(folder_name):
			for matfile in os.listdir(folder_name):
				dataset.append(loadmat(os.path.join(folder_name, matfile)))
	return dataset

if __name__ == "__main__":
	dataset = loadDS()
	random_shuffle(dataset)
	data = dataset[0]
	
	signal = data["sensor_readings"]
	activity = data["activity"]
	print(signal.shape)