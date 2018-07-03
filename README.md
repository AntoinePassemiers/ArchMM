A r c h M M
===========
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

Let's create a simple Gaussian HMM :
```python
from archmm.core import AdaptiveHMM

# Using 5 hidden states and a fully-connected topology
hmm = GHMM(5, arch='ergodic')
```

Then learn from the data using Baum-Welch algorithm :
```python
hmm.fit(data, max_n_iter=20)
```

The data can be either an array of shape (n_samples, n_features)
or a list of such arrays.

Finally evaluate how the model fits a new sequence :
```python
# Using the Akaike Information Criterion
hmm.score(new_data, criterion='aic')
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
- Theano (optional, but required for IO-HMM)

### User installation

Install the package :
```
python setup.py install
```

## Todo

- [ ] Docstrings
- [ ] IO-HMM: change nn backend?
- [ ] Multi-sequence support for HMMs
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