# -*- coding: utf-8 -*-
# test.py
# author : Antoine Passemiers

import numpy as np
from numpy.testing import assert_almost_equal

from archmm.hmm import GHMM


def find_nearest(arr, value, axis=1):
    return np.linalg.norm(np.abs(arr - value), axis=axis).argmin()


def assert_array_almost_equal_by_permuting(A, B, axis=0, **kwargs):
    permuted_A = np.empty(A.shape)
    for i in range(len(B)):
        permuted_A[i] = A[find_nearest(A, B[i], axis=1-axis)]
    assert_array_almost_equal(permuted_A, B, **kwargs)


def test_gmm_multi_sequence():
    hmm = GHMM(3, arch='ergodic')
    X1 = np.random.rand(150, 10)
    X1[:50, 2] += 78
    X1[50:100, 7] -= 98
    X2 = np.random.rand(100, 10)
    X2[:50, 7] -= 98
    X = [X1, X2]
    hmm.fit(X, max_n_iter=10)
    pi = np.sort(hmm.pi)
    a = np.sort(hmm.a.flatten())
    assert_almost_equal(pi, [0., 0.5, 0.5])
    assert_almost_equal(a,
        [0., 0., 0., 0., 0.02, 0.02, 0.98, 0.98, 1.])