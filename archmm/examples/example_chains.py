# -*- coding: utf-8 -*-
# Example with Markov chains

import numpy as np
import matplotlib.pyplot as plt

from archmm.chains import MarkovChainClassifier


if __name__ == "__main__":
    n_classes   = 3
    n_symbols   = 30
    n_sequences = 15
    X = [np.random.randint(0, n_symbols, size = 100) for i in range(n_sequences)]
    y = np.random.randint(0, n_classes, size = n_sequences)

    model = MarkovChainClassifier(n_classes, n_symbols)
    model.fit(X, y)

    predictions = model.classify(X)
    print(y)
    print(predictions)