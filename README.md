[![Travis-CI Build Status](https://travis-ci.org/AntoinePassemiers/ArchMM.svg?branch=master)](https://travis-ci.org/AntoinePassemiers/ArchMM)
# A r c h M M

ArchMM (Architectural Markov Models) is a Cython library for machine learning and data analysis where most of the models and 
algorithms are related to Markov models.
Markov models are descriptive models with strong statistical background. They are very adaptive, quite fast, and only rely on the Markov assumption.
ArchMM offers off-the-shelf HMMs with different topologies (ergodic, linear, etc.), Markov Random Fields for image segmentation,
Input-Output HMMs, and parameter estimation algorithms such as k-means or change point detection algorithms.


Implementations
---------------

- Markov Chain Classifier
- Gaussian Hidden Markov Model (GHMM)
- Multinomial Hidden Markov Model (MHMM)
- Gaussian Markov Random Field (GaussianMRF)

- K-means
- Fuzzy C-Means

Under development
-----------------

- GMM Hidden Markov Model

Debug:

- Input-Output Hidden Markov Model
- Change Point Detection (through polynomial regression)

Design phase:

- CUSUM algorithm
- Fuzzy Hidden Markov Model
- Maximum-Entropy Markov Model
- Multi-Space Distribution Markov Model


How to use it
-------------

Let's create a simple Multinomial HMM,
using 5 hidden states and a fully-connected topology.
```python
from archmm.hmm import MHMM

hmm = MHMM(5, arch='ergodic')
```

Then learn from the data using Baum-Welch algorithm :
```python
hmm.fit(data, max_n_iter=20)
```

The data can be either an array of shape (n_samples, n_features)
or a list of such arrays.

Evaluate how the model fits a new sequence :
```python
# Using the Akaike Information Criterion
# The score value must be as small as possible
score = hmm.score(new_data, criterion='aic')
```

Retrieve the most likely sequence of hidden states:
```python
hidden_sequence = hmm.decode(new_data)
```

Manually set pre-estimated parameters:
```python
hmm = GHMM(11, arch='linear') # Gaussian HMM
hmm.a = a # Transition probabilities
hmm.pi = pi # Initial state probabilities
hmm.mu = mu # Mean vectors
hmm.sigma = sigma # Covariance matrices
```

Generate random samples based on the estimated distribution parameters:
```python
n_samples = 3000
new_data = hmm.sample(n_samples)
```

Full examples can be found in *examples/* folder

Installation
------------

### Dependencies


To get ArchMM to work on your computer, you will need:

- Python
- Numpy (>= 1.6.1)
- Scipy
- Cython

### User installation

Install the package :
```
python setup.py install
```

## Todo

- [x] Automatic doc generation
- [ ] IO-HMM: Implement MLP
- [x] Multi-sequence support for HMMs
- [x] Custom HMM topology
- [ ] Working with missing values
- [ ] Finish implementing GMM-HMM
- [ ] Use nogil blocks where necessary (GHMM and MHMM)
- [ ] Make MRF compatible with different distributions (GMM, multinomial)
- [ ] Fuzzy HMM
- [ ] Maximum-Entropy Markov Model
- [ ] Monte Carlo Markov Chain
- [ ] Markov chains: n-transition probabilities
- [ ] Implement MSD-HMM (for fun)
- [ ] Automatic setup.py scripts that recursively look for *.c, *.py and *.pyx files
- [ ] Visualize models using matplotlib and networkx