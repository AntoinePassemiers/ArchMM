A r c h M M
===========
ArchMM (Architectural Markov Models) is a Cython library for machine learning and data analysis where most of the algorithms are related to Markov models.
Markov models are descriptive models with strong statistical background. They are very adaptive, quite fast, and rely only on the Markov assumption.
ArchMM proposes the implementation of different HMM topologies (ergodic, linear, Bakis, cyclic), exotic Markov model variants, 


Implementations
---------------

- Markov Chain Classifier
- Gaussian Hidden Markov Model
- Markov Random Field
- Input-Output Hidden Markov Model

- K-means
- Fuzzy C-Means
- Change Point Detection (through regression)

Under development
-----------------

- Fuzzy Hidden Markov Model

Design phase:

- Maximum-Entropy Markov Model
- Multi-Space Distribution Markov Model


How to use it
-------------

Let's create a regular HMM :
```python
from archmm.core import AdaptiveHMM

# Using 5 hidden states, a fully-connected topology and normalizing the inputs
# Also setting 0 as being a missing value
hmm = AdaptiveHMM(5, "ergodic", standardize = True, missing_value = 0)
```

Then learn from the data using the Baum-Welch algorithm :
```python
hmm.fit(data)
```

Finally evaluate how the model fits a new sequence :
```python
# Using the Akaike Information Criterion
hmm.score(data_2, mode = "aic")
```

To instanciate an IO-HMM, there are more parameters to specify :
```python
config = IOConfig()
config.architecture = "linear"  # Linear topology
config.n_iterations = 20        # Number of iterations of the GEM
config.s_learning_rate  = 0.006 # Learning rate of the initial state unit
config.o_learning_rate  = 0.01  # Learning rate of the state transition units
config.pi_learning_rate = 0.01  # Learning rate of the output units

# Number of hidden neurons and number of epochs (when using MLP only)
config.s_nhidden  = 5
config.o_nhidden  = 4
config.pi_nhidden = 4
config.pi_nepochs = 5
config.s_nepochs  = 6
config.o_nepochs  = 5

# Instanciate IOHMM with 5 hidden states
iohmm = AdaptiveHMM(5, has_io = True, standardize = False)
```
Only classification is supported yet. To train an IOHMM for classification, we make use
of the Generalized Expectation-Maximization algorithm.
```python
fit = iohmm.fit(X, targets = y, n_classes = 2, is_classifier = True, parameters = config)
```

Now we can classify our new data :
```python
prediction = iohmm.predictIO(validation_X)
```
Installation
------------

### Dependencies


To get ArchMM to work on your computer, you will need:

- Python 2.7
- Numpy (>= 1.6.1)
- Scipy
- Cython
- Theano (optional)
- cvxpy (optional)

### User installation

Install the package :
```
python setup.py install
```
