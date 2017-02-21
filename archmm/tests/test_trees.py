# -*- coding: utf-8 -*-

from archmm.tests.utils import *
from archmm.trees.tree import *

data = np.asarray(np.array([
	[0, 0, 0], # 0
	[0, 0, 1], # 0
	[1, 0, 0], # 1
	[2, 0, 0], # 1
	[2, 1, 0], # 1
	[2, 1, 1], # 0
	[1, 1, 1], # 1
	[0, 0, 0], # 0
	[0, 1, 0], # 1
	[2, 1, 0], # 1
	[0, 1, 1], # 1
	[1, 0, 1], # 1
	[1, 1, 0], # 1
	[2, 0, 1]  # 0
]), dtype = np.double)
targets = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

udata = np.asarray(np.array([
	[0, 0, 1], # 1
	[0, 0, 0], # 1
]), dtype = np.double)
utargets = np.array([1, 1])
probs = np.array([2.0/3, 0.5, 0, 0, 0, 1, 0, 2.0/3, 0, 0, 0, 0, 0, 1])

def test_id3_predictions():
	tree = ClassificationTree()
	tree.applyID3(data, targets)
	y = tree.classify(data)
	assert_array_equal(targets, np.argmax(y, axis = 1))
	tree.update(udata, utargets)
	y = tree.classify(data)
	assert_array_almost_equal(probs, y[:, 0])

def test_id3_with_longs():
	long_targets = np.asarray(targets, dtype = np.long)
	tree = ClassificationTree()
	tree.applyID3(data, long_targets)
	y = tree.classify(data)
	assert_array_equal(long_targets, np.argmax(y, axis = 1))

@dec.slow
def test_big_tree():
	n_instances = 100
	X = np.asarray(np.random.rand(n_instances, 25), dtype = np.double)
	y = np.random.randint(0, 10, size = n_instances)
	tree = ClassificationTree(max_nodes = 1000, partitioning = 2)
	tree.applyID3(X, y)