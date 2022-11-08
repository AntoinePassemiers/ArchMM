import numpy as np

from archmm.hmm import HMM
from archmm.distributions import MultivariateGaussian


sequences = []
sequence = np.random.rand(1800, 3)
sequence[1200:, :] += 0.5
sequences.append(sequence)
sequence = np.random.rand(1800, 3)
sequence[300:, :] += 0.5
sequences.append(sequence)

model = HMM()
for _ in range(3):
    model.add_state(MultivariateGaussian(3))
model.fit(sequences)

print(list(model.decode(sequences[0])))
print(list(model.decode(sequences[1])))

print(f'Log-likelihood: {model.score(sequences)}')
