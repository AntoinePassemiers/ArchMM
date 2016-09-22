# -*- coding: utf-8 -*-

import sys, pickle
import numpy as np

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from ArchMM.Cyfiles.HMM_Core import *




if __name__ == "__main__":
    
    data = np.arange(1, 1001).reshape(500, 2)
    data[59, 1] = 78
    hmm = AdaptiveHMM(10)
    hmm.fit(data, dynamic_features = False)
    print(hmm.getMu())
    
    """
    data = pickle.load(open("data_hmm_input", 'rb'))
    data = data[:, -2:]
    data = np.array(fillMissingValues(data[:, :]))
    hmm = AdaptiveHMM(10)
    hmm.fit(data, dynamic_features = False)
    print(hmm.getMu())
    # print("AIC  : %f" % hmm.score(data, mode = CRITERION_AIC))
    # print("AICc : %f" % hmm.score(data, mode = CRITERION_AICC))
    # hmm.randomSequence(45)
    """
    print("Finished")
