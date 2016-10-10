# -*- coding: utf-8 -*-

import sys, pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from ArchMM.Cyfiles.HMM_Core import *


def testHMM():
    data = np.arange(1, 1001, dtype = np.double).reshape(500, 2)
    data[59, 1] = 78
    hmm = AdaptiveHMM(10, "ergodic", standardize = True, missing_value = 0)
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
    
    plt.plot(data3[:, 1])
    plt.show()
    
def villoTest():
    data = [11,11,10,10,9,9,9,11,10,10,9,10,11,12,
            11,11,13,14,13,14,16,18,17,18,17,18,17,
            17,17,16,15,16,16,16,16,17,17,17,17,12,
            12,12,12,11,12,12,12,13,14,14,14,15,15,
            15,15,15,15,15,15,15,14,14,14,14,14,14,
            3,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,2,2,1,
            2,1,3,5,6,5,4,5,4,5,5,5,6,9,8,8,8,8,9,9,
            9,9,8,8,8,9,10,9,9,10,9,9,8,7,7,7,7,8,6,
            6,5,4,3,2,3,2,3,1,3,2,4,5,4,5,7,8,8,9,8,
            9,8,8,7,7,7,7,7,7,8,
            7,7,8,9,9,9,8,9,9,9,8,8,8,6,7,7,7,7,7]
    f = np.array(data)[1:] - np.array(data)[:-1]
    D = np.zeros((len(data), 2))
    D[:, 0] = np.array(data)[:]
    hmm = AdaptiveHMM(30, "linear", standardize = False)
    hmm.fit(D, dynamic_features = False, n_iterations = 100)
    states, seq = hmm.randomSequence(len(data))
    """
    internal_memory = 11
    for i in range(len(seq)):
        internal_memory += seq[i]
        seq[i] = internal_memory
    """
    print(states)
    print(hmm.getMu()[:])
    print(hmm.getA()[:])
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.step(range(len(data)), data)
    ax2 = fig.add_subplot(222)
    ax2.step(range(len(data)), seq[:, 0])
    plt.show()
    
    
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
    hmm.fit([U], targets = [Y], n_iterations = 5, n_classes = 2)
    
def testMLP():
    X = np.array([[1, 0], [0, 1], [2, 1], [0, 2]])
    Y = np.array([1, 0, 1, 0])
    mlp = MLP(2, 4, 2)
    mlp.train(X, Y)
    print(mlp.predict(X))
    

if __name__ == "__main__":
    villoTest()
    print("Finished")
