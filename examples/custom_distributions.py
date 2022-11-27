import numpy as np

from archmm import HiddenState, Architecture
from archmm.distributions.base import BaseDistribution
from archmm.hmm import HMM


class Bernoulli(BaseDistribution):

    def __init__(self):
        super().__init__()
        self.p: float = float(np.random.rand())

    def _param_update(self, data: np.ndarray, gamma: np.ndarray):
        denominator = np.sum(gamma)
        self.p = float(np.sum(data * gamma) / denominator)

    def _log_pdf(self, data: np.ndarray) -> np.ndarray:
        return np.log(data * self.p + (1. - data) * (1. - self.p))

    def sample(self, n: int) -> np.ndarray:
        return (np.random.rand(n) < self.p).astype(int)


sequence = np.empty(200, dtype=int)
sequence[:70] = np.random.rand(70) < 0.15
sequence[70:] = np.random.rand(130) < 0.87

model = HMM()
states = [HiddenState(Bernoulli()) for _ in range(2)]
Architecture.ergodic(states)
model.add_states(states)
model.fit(sequence)

print(f'Hidden states: {model.decode(sequence)}')
print(f'p parameter for state 0: {states[0].p}')
print(f'p parameter for state 1: {states[1].p}')
