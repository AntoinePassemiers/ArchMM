# -*- coding: utf-8 -*-
#@PydevCodeAnalysisIgnore

import sys
import numpy as np

sys.path.insert(0, '../..')
from Structs import *


if __name__ == "__main__":
    alpha = Flattened_Markov_tensor3(100, 100)
    beta  = Flattened_Markov_tensor3(100, 100)
    alpha[5, 5] = 78
    beta[5, 5] = 100
    gamma = alpha + beta
    print(gamma[5, 5])
    alpha[45, 45] = 789
    print(alpha[45, 45])