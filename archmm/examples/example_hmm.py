# -*- coding: utf-8 -*-
# Example with Hidden Markov Models

import numpy as np
import matplotlib.pyplot as plt

from archmm.core import HMM


if __name__ == "__main__":
	signal = np.empty(1200, dtype = np.float)
	signal[:300] = np.random.normal(4.0, 0.7, 300)
	signal[300:500] = np.random.normal(-1.1, 1.2, 200)
	signal[500:600] = np.random.normal(2.4, 0.2, 100)
	signal[600:1000] = np.random.normal(0.2, 2.0, 400)
	signal[1000:1200] = np.random.normal(0.2, 0.15, 200)

	signal = signal[::2]

	# signal = np.asarray([signal, signal]).T

	hmm = HMM(5, "ergodic")
	hmm.fit(signal, n_iterations = 10)
	d = hmm.decode(signal)

	print(list(d))

	plt.plot(signal)
	plt.show()