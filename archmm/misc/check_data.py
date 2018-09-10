# -*- coding: utf-8 -*-
# check_data.py
# author: Antoine Passemiers

import numpy as np
import copy
import collections

from archmm.exceptions import BadInputData


def is_iterable(obj):
	return isinstance(obj, collections.Iterable)


def check_hmm_sequence(X):
	X = np.squeeze(np.asarray(X))
	if len(X.shape) == 1:
		n_features = 1
	elif len(X.shape) == 2:
		n_features = X.shape[1]
	else:
		raise BadInputData('A HMM sequence must be at most 2-dimensional')
	return X, n_features


def check_hmm_sequences_list(X_s):
	if is_iterable(X_s):
		if isinstance(X_s, np.ndarray) and len(X_s.shape) < 3:
			X_s = [X_s]
		if len(X_s) > 0:
			if is_iterable(X_s[0]):
				X_s = copy.copy(X_s) # Shallow copy
				tmp = list()
				for i in range(len(X_s)):
					X_s[i], n_features = check_hmm_sequence(X_s[i])
					tmp.append(n_features)
				if not all(n_features == tmp[0] for n_features in tmp):
					raise BadInputData('Number of features must be kept ' + \
						'from one sequence to another')
			else:
				n_features = 1
				X_s = [np.asarray(X_s)]
		else:
			raise BadInputData('Found an empty sequence as input')
	else:
		raise BadInputData('Expected an iterable as input')
	return X_s, n_features
