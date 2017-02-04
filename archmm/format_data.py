# -*- coding: utf-8 -*-


import numpy as np

SEQUENCE_ELEM_T = np.double

class DataWrapper:
	def __init__(self, data):
		data = np.asanyarray(data, order = 'C')
		self.data = data.view(dtype = SEQUENCE_ELEM_T)
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
		print(self.shape)
		self.ndim = len(self.shape)
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