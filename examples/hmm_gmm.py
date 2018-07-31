import numpy as np

from archmm.hmm import GMMHMM


if __name__ == '__main__':

    hmm = GMMHMM(3, arch='ergodic', n_components=3)

    X = np.random.rand(180, 10)

    X[:20, 5] += 4
    X[20:40, 6] -= 2
    X[40:60, 0] += 3
    X[:60, 2] += 78

    X[60:80, 1] += 1
    X[80:100, 4] -= 3
    X[100:120, 6] += 2
    X[60:120, 7] -= 98

    X[120:140, 0] -= 4
    X[140:160, 1] += 3
    X[160:, 3] += 1



    hmm.fit(X, max_n_iter=20)

    print(hmm)

    print(hmm.mu)
    print(hmm.decode(X))
    print(hmm.score(X))