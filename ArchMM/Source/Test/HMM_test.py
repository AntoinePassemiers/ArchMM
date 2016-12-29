# -*- coding: utf-8 -*-
#@PydevCodeAnalysisIgnore

import sys, pickle
import numpy as np
# import matplotlib.pyplot as plt

import theano

sys.path.insert(0, '../../Source')
sys.path.insert(0, '../..')
from HMM_Core import *


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
    hmm = AdaptiveHMM(50, "linear", standardize = False)
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
    U = np.arange(1, 51).reshape(25, 2)
    U[5, 1] = 78
    U[9, 1] = 0
    U[12, 1] = 78
    U[18, 1] = -4586
    U2 = np.random.rand(25, 2)
    U3 = np.arange(1, 51).reshape(25, 2)
    U4 = np.random.rand(25, 2)
    U5 = np.arange(51, 101).reshape(25, 2)
    U6 = np.random.rand(25, 2)
    
    Y = np.ones(25, dtype = np.int32)
    Y2 = np.zeros(25, dtype = np.int32)
    Y3 = np.ones(25, dtype = np.int32)
    Y4 = np.zeros(25, dtype = np.int32)
    Y5 = np.ones(25, dtype = np.int32)
    Y6 = np.zeros(25, dtype = np.int32)
    
    config = IOConfig()
    config.architecture = "linear"
    config.n_iterations = 20
    config.s_learning_rate = 0.006
    config.o_learning_rate = 0.01
    config.pi_learning_rate = 0.01
    config.s_nhidden = 4
    config.o_nhidden = 4
    config.pi_nhidden = 4
    config.pi_nepochs = 5
    config.s_nepochs = 5
    config.o_nepochs = 5
    hmm = AdaptiveHMM(5, has_io = True, standardize = False)
    inputs  = [U, U2, U3, U4, U5, U6]
    outputs = [Y, Y2, Y3, Y4, Y5, Y6]
    fit = hmm.fit(inputs, targets = outputs, 
                  dynamic_features = False,
                  n_classes = 2, is_classifier = True, parameters = config)
    """
    print(fit[0])
    print(fit[1])
    print(fit[2])
    print(fit[3])
    for i in range(4):
        np.save(open("iohmm_training_%i" % i, "wb"), fit[i])
    """

    U7  = np.arange(7, 57).reshape(25, 2)
    U8  = np.random.rand(25, 2)
    U9  = np.arange(3, 53).reshape(25, 2)
    U10 = np.random.rand(25, 2)
    U11 = np.arange(31, 81).reshape(25, 2)
    U12 = np.random.rand(25, 2)
    U13 = np.arange(18, 68).reshape(25, 2)
    U14 = np.random.rand(25, 2)
    
    print(hmm.predictIO(U)[0])
    print(hmm.predictIO(U2)[0])
    print(hmm.predictIO(U3)[0])
    print(hmm.predictIO(U4)[0])
    print(hmm.predictIO(U5)[0])
    print(hmm.predictIO(U6)[0])
    print(hmm.predictIO(U7)[0])
    print(hmm.predictIO(U8)[0])
    print(hmm.predictIO(U9)[0])
    print(hmm.predictIO(U10)[0])
    print(hmm.predictIO(U11)[0])
    print(hmm.predictIO(U12)[0])
    print(hmm.predictIO(U13)[0])
    print(hmm.predictIO(U14)[0])
    
def testIOHMMSimulation():
    U = np.arange(1, 51).reshape(25, 2)
    Y = np.arange(1, 51).reshape(25, 2) + 25
    hmm = AdaptiveHMM(7, has_io = True)
    print(hmm.fit([U], targets = [Y], n_iterations = 5, n_epochs = 1, is_classifier = False))


if __name__ == "__main__":
    testIOHMM()
    # villoTest()
    print("Finished")