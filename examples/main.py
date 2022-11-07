import numpy as np

from archmm.hmm import HMM, GaussianHMM

sequence = np.random.rand(1800, 3)
sequence[1200:, :] += 0.5

model = GaussianHMM(3, 3)
model.fit(sequence)

print(list(model.decode(sequence)))

print('Finished')
