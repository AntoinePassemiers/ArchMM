# -*- coding: utf-8 -*-

import numpy as np

from archmm.svm.kernel import *
from archmm.utils import *

if USE_CVXPY:
    import cvxpy

# Soft-margin and hard-margin SVMs
# TODO : use sparse matrices to store the labels

KERNEL_DATA_T = np.double

def binarizeLabels(labels, n_classes):
    n_instances = len(labels)
    new_labels = np.zeros((n_instances, n_classes), dtype = np.int)
    new_labels[np.arange(n_instances), labels] = 1
    return new_labels

class SoftMarginSVM:
    @requiresCvxPy(True)
    def __init__(self, state_id, n_instances, n_classes, gamma = 1.0):
        self.state_id = state_id
        self.n_instances = n_instances
        self.n_classes = n_classes
        self.kernel = None
        self.gamma = gamma
        self.U = None

    @todo
    def processOutput(self, X):
        pass
    
    @todo
    def __getstate__(self):
        pass
    
    @todo
    def __setstate__(self, state):
        pass
    
    def train(self, train_set_x, target_set, memory_array, weights, is_mv):
        self.kernel = linear_kernel(train_set_x, train_set_x)
        E = binarizeLabels(target_set, self.n_classes)
        U = cvxpy.Variable(self.n_instances, self.n_classes)
        objective = cvxpy.Maximize(- 0.5 * cvxpy.trace(U.T * self.kernel * U) + cvxpy.trace(E.T * U))
        constraints = [
            U <= self.gamma * E
        ]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve()
        self.U = U

n_instances, n_features, n_classes = 200, 30, 5
X = np.random.rand(n_instances, n_features)
y = np.random.randint(0, n_classes, size = n_instances)
svm = SoftMarginSVM(0, n_instances, n_classes)
svm.train(X, y, None, None, None)
