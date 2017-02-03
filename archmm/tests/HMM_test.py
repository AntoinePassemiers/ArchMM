# -*- coding: utf-8 -*-

import sys, pickle
import numpy as np

import theano

sys.path.insert(0, '../../ArchMM/Source')
sys.path.insert(0, '..')
from HMM_Core import *
from ChangePointDetection import *

import matplotlib.pyplot as plt


def testCPD():
    data = np.random.random((150, 2))
    data[26, 0] = 500.0
    data[42, 0] = 500.0
    data[91, 1] = 500.0
    cusum = GTD(cost_func = 4)
    cusum.detectPoints(data)

def testUnivariateHMM():
    data = np.arange(1, 501, dtype = np.double).reshape(500, 1)
    data[59, 0] = -78
    hmm = AdaptiveHMM(10, "ergodic", standardize = False, missing_value = 0)
    hmm.fit(data, dynamic_features = False)
    print(hmm.getMu())
    data2 = np.arange(1, 501, dtype = np.double).reshape(500, 1)
    data2[259, 0] = 78
    data2[357, 0] = 78
    data3 = np.arange(1, 501, dtype = np.double).reshape(500, 1)
    data3[25, 0]  = 78
    data3[259, 0] = 0
    data3[370, 0] = 78
    data3[159, 0] = -4586
    data3[111, 0]  = 884
    print(hmm.score(data, mode = "aicc"))
    print(hmm.score(data2, mode = "aicc"))
    print(hmm.score(data3, mode = "aicc"))

def testHMM():
    data = np.arange(1, 1001, dtype = np.double).reshape(500, 2)
    data[59, 1] = -78
    hmm = AdaptiveHMM(10, "ergodic", standardize = False, missing_value = 0)
    hmm.fit(data, dynamic_features = False)
    print(hmm.getMu())
    data2 = np.arange(1, 1001).reshape(500, 2)
    data2[259, 1] = 78
    data2[357, 1] = 78
    data3 = np.arange(1, 1001).reshape(500, 2)
    data3[25, 1]  = 78
    data3[259, 1] = 0
    data3[370, 1] = 78
    data3[159, 1] = -4586
    data3[111, 1]  = 884
    print(hmm.score(data, mode = "aicc"))
    print(hmm.score(data2, mode = "aicc"))
    print(hmm.score(data3, mode = "aicc"))
    
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
    # testUnivariateHMM()
    testHMM()
    # testCPD()
    # testIOHMM()
    # villoTest()
    print("Finished")