import numpy as np

from archmm.hmm import GMMHMM


if __name__ == '__main__':

    hmm = GMMHMM(3, arch='ergodic', n_components=3)

    X = np.random.rand(150, 10)

    X[:50, 2] += 78
    X[50:100, 7] -= 98

    hmm.fit(X, max_n_iter=20)

    print(hmm)

    print(hmm.mu)
    print(hmm.decode(X))
    print(hmm.score(X))