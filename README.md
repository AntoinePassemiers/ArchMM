[![Travis-CI Build Status](https://travis-ci.org/AntoinePassemiers/ArchMM.svg?branch=master)](https://travis-ci.org/AntoinePassemiers/ArchMM)
# ArchMM

Flexible Cython library for designing Hidden Markov Models

```python
from archmm.hmm import HMM
from archmm.distributions.gaussian import MultivariateGaussian

# Create a HMM with 3 hidden states
# and 4-dimensional multivariate Gaussian distributions
model = HMM()
for _ in range(3):
    model.add_state(MultivariateGaussian(4))

# Train the model with the Baum-Welch algorithm.
# `sequences` is either a NumPy array or a list of NumPy arrays.
# If a list, elements are treated as different sequences.
# Arrays can have arbitrary dimensions, as long as it complies
# with the support of the distributions. In the present case,
# arrays have shape (n, 4), and n is sequence-dependent.
model.fit(sequences)
```

Custom distributions can be defined as hidden states.
More examples can be found in the `examples` folder.

### Install the library

Please make sure you have Cython installed first.
Then you can build the library simply with:

```
python setup.py install
```

### Ongoing work
- Implement sample method
- Implement score method
- Improved initialization
- NaN support
- Arbitrary topologies
- GMMHMM
- Input-Output HMM
- MRF, etc.
