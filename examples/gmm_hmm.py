import numpy as np

from archmm import HiddenState, Architecture
from archmm.distributions.mixture import Mixture

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
states = []
for _ in range(2):
    states.append(HiddenState(Mixture(
        MultivariateGaussian(3),
        MultivariateGaussian(3),
        MultivariateGaussian(3)
    )))
Architecture.ergodic(states)
model.add_states(states)

model.fit(sequences)

print(list(model.decode(sequences[0])))
print(list(model.decode(sequences[1])))

print(f'Log-likelihood: {model.score(sequences)}')
