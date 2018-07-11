# -*- coding: utf-8 -*-
# hmm_gaussian.py
# author: Antoine Passemiers

import numpy as np

from archmm.hmm import GHMM


if __name__ == '__main__':

    # Gaussian HMM with 3 hidden states and
    # ergodic topology
    hmm = GHMM(3, arch='ergodic')

    # Sequence X1 (150 samples):
    # 50 samples in state 0 +
    # 50 samples in state 1 +
    # 50 samples in state 2
    X1 = np.random.rand(150, 10)
    X1[:50, 2] += 78
    X1[50:100, 7] -= 98

    # Sequence X2 (100 samples):
    # 50 samples in state 1 +
    # 50 samples in state 2
    X2 = np.random.rand(100, 10)
    X2[:50, 7] -= 98
    X = [X1, X2]

    # Fit the 2 sequences
    hmm.fit(X, max_n_iter=3)

    # Display summary
    print(hmm)

    # The two sequences started in different states
    # We expect the start probabilities to be a permutation
    # of the following probability vector: [0.5, 0.5, 0.0]
    print("\nState start probabilities:")
    print(hmm.pi)
    print("\nState transition probabilities:")
    print(hmm.a)
    print("\nMean vectors:")
    print(hmm.mu)

    # Let's decode the first sequence and see if the
    # most likely sequence of hidden states is of the
    # form: [a a a a ... b b b b ... c c c c],
    # where a, b, c are in {0, 1, 2}
    print("\nSequence X1 decoded:")
    print(hmm.decode(X1))
    print("\nAIC: %s" % str(hmm.score(X1)))