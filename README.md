A r c h M M
===========
ArchMM (Architectural Markov Models) is a Cython library for machine learning where most of the algorithms are based on the Markov assumption.
It contains the implementation of different HMM topologies (ergodic, linear, Bakis, cyclic), and adaptations of popular machine learning
algorithms used under the Markov hypothesis (such as the well-known Multi-Layer Perceptron, adapted to the IOHMM).

The Input-Output Hidden Markov Model (IO-HMM) is a markovian model where both output values and transition probabilities are computed using 
sub-models such as multi-layered perceptrons. The learning algorithm is based on the Generalized Expectation-Maximization procedure.

ArchMM provides also clustering algorithms, change point detection algorithms, and fuzzy HMMs.

How to use it
-------------

Let's create a regular HMM :
```python
from HMM_Core import AdaptiveHMM

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
- Theano

### User installation

Install the package :
```
python setup.py install
```
If you have Cython, the C files will be re-generated before to be compiled.
