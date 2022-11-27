from archmm.hmm import HMM

from archmm import HiddenState, Architecture
from archmm.distributions import MultivariateGaussian


# 3-states HMM
model = HMM()
states = [HiddenState(MultivariateGaussian(3)) for _ in range(3)]
model.add_states(states)

# Define linear architecture explicitly: state 1 -> state 2 -> state 3
states[0].allow_start()
states[0].can_transit_to(states[0])
states[0].can_transit_to(states[1])
states[1].can_transit_to(states[1])
states[1].can_transit_to(states[2])
states[2].can_transit_to(states[2])

# Define linear architecture with helper function
Architecture.linear(states)
