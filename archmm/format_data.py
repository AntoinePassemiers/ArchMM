# -*- coding: utf-8 -*-

import numpy as np
import ctypes

from archmm.utils import *

PYTHON_32_BITS_MODE = 32
PYTHON_64_BITS_MODE = 64
PYTHON_XX_BITS_MODE = ctypes.sizeof(ctypes.c_voidp) * 8

SEQUENCE_ELEM_T = np.double
INT_ELEM_T = np.int

class TimeSeries3D:
	def __init__(self):
		self.concat_to_one_series = None
		self.mu = None
		self.sigma = None
	@abstractmethod
	def getData():
		return
	def ensureConcatToOneSeries(self):
		if not self.concat_to_one_series:
			data = self.getData()
			assert(len(data.shape) == 3)
			self.concat_to_one_series = np.concatenate(data)
	def getMu(self):
		if not self.concat_to_one_series:
			self.ensureConcatToOneSeries()
		if not self.mu:
			self.mu = np.mean(axis = 0)
		return self.mu
	def getSigma(self):
		if not self.concat_to_one_series:
			self.ensureConcatToOneSeries()
		if not self.sigma:
			self.sigma = np.cov(axis = 0)
		return self.sigma

class DataWrapper(TimeSeries3D):
	def __init__(self, data, ndim = 3):
		self.ndim = ndim
		data = np.asanyarray(data, order = 'C')
		if data.dtype.type is np.string_:
			raise ArrayTypeError(
				"Strings are not supported, "
				"please use a label encoder first."
			)
		try:
			self.data = data.view(dtype = SEQUENCE_ELEM_T)
		except:
			self.data = np.asarray(data, dtype = SEQUENCE_ELEM_T)
		if len(self.data.shape) == 2:
			nrows, ncols = self.data.shape
			nbytes = self.data.itemsize
			# Changing strides without copying
			self.data = np.lib.stride_tricks.as_strided(
				self.data,
				shape = (1, nrows, ncols),
				strides = (nrows * ncols * nbytes, ncols * nbytes, nbytes)
			)
		assert(np.may_share_memory(data, self.data))
		self.shape = self.data.shape
		print("Array shape : " + str(self.shape))
		assert(self.ndim == len(self.shape)) # TODO
		assert(self.ndim == 3)
	def __getitem__(self, key):
		return self.data[key]
	def __setitem__(self, key, value):
		self.data[key] = value

def format_data(data):
	if isinstance(data, DataWrapper):
		return data
	else:
		return DataWrapper(data)

def ensure_array(data, ndim = 1, msg = str()):
	data = np.asanyarray(data, order = 'C')
	if data.ndim != ndim:
		raise DataDimensionError(msg)