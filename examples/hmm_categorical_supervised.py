import numpy as np

from archmm.hmm import MHMM


if __name__ == '__main__':

    hmm = MHMM(3, arch='ergodic')

    X = np.empty(150)
    X[:50] = np.random.choice(np.arange(3), size=50, p=[0.1, 0.7, 0.2])
    X[50:100] = np.random.choice(np.arange(3), size=50, p=[0.5, 0.2, 0.3])
    X[100:150] = np.random.choice(np.arange(3), size=50, p=[0.4, 0.1, 0.5])

    Y = np.empty(150, dtype=np.int)
    Y[:50] = 2
    Y[50:100] = 0
    Y[100:150] = 1

    hmm.fit(X, y=Y, max_n_iter=50)

    print(hmm)

    print("\nState start probabilities:")
    print(hmm.pi)
    print("\nState transition probabilities:")
    print(hmm.a)

    print("\nEmission probabilities:")
    print(hmm.proba)

    print(hmm.decode(X))

    print(hmm.score(X, criterion='aic'))