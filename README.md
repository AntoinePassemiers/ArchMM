A r c h M M
===========
ArchMM (Architectural Markov Models) is a Cython library for machine learning, based on Hidden Markov Models. 
It contains the implementation of different HMM topologies (ergodic, linear, Bakis, cyclic), and adaptations of popular machine learning
algorithms used under the Markov hypothesis (alternatives of the Input-Output Hidden Markov Model). 

The Input-Output Hidden Markov Model (IO-HMM) is a markovian model where both output values and transition probabilities are computed using 
sub-models such as multi-layered perceptrons. The learning algorithm is based on the Generalized Expectation-Maximization procedure.

Installation
------------

### Dependencies


To get ArchMM to work on your computer, you will need:

- Python 2.7
- Numpy (>= 1.6.1)
- Scipy
- Theano
- Cython

### User installation


The setup file is not ready yet, you will have to compile the project yourself using Cython :(
