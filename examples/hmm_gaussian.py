import numpy as np

from archmm.hmm import GHMM


if __name__ == '__main__':

    hmm = GHMM(3, arch='ergodic')

    X = np.random.rand(150, 10)

    X[:50, 2] += 78
    X[50:100, 7] -= 98

    hmm.fit(X, max_n_iter=3)

    print(hmm)

    print(hmm.pi)
    print(hmm.a)
    print(hmm.mu)
    print(hmm.decode(X))
    print(hmm.score(X))