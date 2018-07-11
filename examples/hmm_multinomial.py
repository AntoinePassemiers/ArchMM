import numpy as np

from archmm.hmm import MHMM


if __name__ == '__main__':

    # Only ergodic topology is supported for the moment
    # Performance will be much higher on this example
    # once left-to-right topologies will be fully implemented
    hmm = MHMM(3, arch='ergodic')

    X = np.empty(150)

    X[:50] = np.random.choice(np.arange(3), size=50, p=[0.1, 0.7, 0.2])
    X[50:100] = np.random.choice(np.arange(3), size=50, p=[0.5, 0.2, 0.3])
    X[100:150] = np.random.choice(np.arange(3), size=50, p=[0.4, 0.1, 0.5])

    hmm.fit(X, max_n_iter=20)

    print(hmm)

    print("\nState start probabilities:")
    print(hmm.pi)
    print("\nState transition probabilities:")
    print(hmm.a)

    print("\nEmission probabilities:")
    print(hmm.proba)

    print(hmm.decode(X))

    print(hmm.score(X, criterion='aic'))