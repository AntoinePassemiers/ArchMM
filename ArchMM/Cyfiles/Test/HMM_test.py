# -*- coding: utf-8 -*-

import sys, pickle
import numpy as np

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from ArchMM.Cyfiles.HMM_Core import *


def testHMM():
    data = np.arange(1, 1001, dtype = np.double).reshape(500, 2)
    data[59, 1] = 78
    hmm = AdaptiveHMM(10, "linear", standardize = True, missing_value = 0)
    hmm.fit(data, dynamic_features = False)
    print(hmm.getMu())
    data2 = np.arange(1, 1001).reshape(500, 2)
    data2[259, 1] = 78
    data2[357, 1] = 78
    data3 = np.arange(1, 1001).reshape(500, 2)
    data3[25, 1] = 78
    data3[259, 1] = 0
    data3[370, 1] = 78
    data3[159, 1] = -4586
    data[111, 1] = 884
    print(hmm.score(data, mode = "probability"))
    print(hmm.score(data2, mode = "probability"))
    print(hmm.score(data3, mode = "probability"))
    
def testHMMwithMissingValues():
    data = pickle.load(open("data_hmm_input", 'rb'))
    data = data[:, -2:]
    # data = np.array(fillMissingValues(data[:, :]))
    hmm = AdaptiveHMM(10)
    hmm.fit(data, dynamic_features = False)
    print(hmm.getMu())
    # print("AIC  : %f" % hmm.score(data, mode = CRITERION_AIC))
    # print("AICc : %f" % hmm.score(data, mode = CRITERION_AICC))
    # hmm.randomSequence(45)
    
def testIOHMM():
    U = np.arange(1, 1001).reshape(500, 2)
    U[25, 1] = 78
    U[259, 1] = 0
    U[370, 1] = 78
    U[159, 1] = -4586
    Y = np.random.randint(2, size = (500))
    hmm = AdaptiveHMM(10, has_io = True)
    hmm.fit([U], targets = [Y], n_iterations = 70, n_classes = 2)
    
def testMLP():
    X = np.array([[1, 0], [0, 1], [2, 1], [0, 2]])
    Y = np.array([1, 0, 1, 0])
    mlp = MLP(2, 4, 2)
    mlp.train(X, Y)
    print(mlp.predict(X))
    

if __name__ == "__main__":
    testIOHMM()
    print("Finished")
