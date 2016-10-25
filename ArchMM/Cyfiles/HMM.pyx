# -*- coding: utf-8 -*-

import ctypes, pickle
import numpy as np
from numpy.random import randn, random, dirichlet
from scipy.spatial.distance import cdist # TO REMOVE
from cpython cimport array 
from libc.stdio cimport *
cimport libc.math

include "Artifacts.pyx"
include "KMeans.pyx"
include "ChangePointDetection.pyx"
include "MLP.pyx"

theano_support = False


cpdef unsigned int ARCHITECTURE_LINEAR = 1
cpdef unsigned int ARCHITECTURE_BAKIS = 2
cpdef unsigned int ARCHITECTURE_LEFT_TO_RIGHT = 3
cpdef unsigned int ARCHITECTURE_ERGODIC = 4
cpdef unsigned int ARCHITECTURE_CYCLIC = 5

cpdef unsigned int CRITERION_AIC = 101
cpdef unsigned int CRITERION_AICC = 102
cpdef unsigned int CRITERION_BIC = 103
cpdef unsigned int CRITERION_LIKELIHOOD = 104
cpdef unsigned int CRITERION_NEG_LIKELIHOOD = 105

cpdef unsigned int DISTRIBUTION_GAUSSIAN = 201
cpdef unsigned int DISTRIBUTION_MULTINOMIAL = 202

cpdef unsigned int USE_LOG_IMPLEMENTATION = 301
cpdef unsigned int USE_LIN_IMPLEMENTATION = 302
    


""" Extended versions of the ln, exp, and log product functions 
to prevent the Baum-Welch algorithm from causing overflow/underflow """

LOG_ZERO = np.nan_to_num(- np.inf)

@np.vectorize
def elog(x):
    """ Vectorized version of the extended logarithm """
    return np.log(x) if x != 0 else LOG_ZERO

@np.vectorize
def eexp(x):
    """ Vectorized version of the extended exponential """
    return np.exp(x) if x != LOG_ZERO else 0

cdef ieexp2d(M):
    """ Extended exponential for 2d arrays (inplace operation) """
    cdef Py_ssize_t i, j
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i][j] = libc.math.exp(M[i][j]) if M[i][j] != LOG_ZERO else 0.0
    return M

ctypedef double signal_t
ctypedef cnp.double_t data_t


cdef GaussianGenerator(mu, sigma, n = 1):
    """ Random variables from a gaussian distribution using
    mean [[mu]] and variance-covariance matrix [[sigma]]
    
    Parameters
    ----------
    mu : mean of the input signal
    sigma : variance-covariance matrix of the input signal
    n : number of samples to generate
    Returns
    -------
    Array containing the random variables
    """
    cdef bint has_non_positive_definite_minor = True
    cdef double mcv = 0.00001 # TODO : Use standard deviations in place of a single constant
    cdef Py_ssize_t i, ndim = len(mu)
    r = randn(n, ndim)
    if n == 1:
        r.shape = (ndim,)
    while has_non_positive_definite_minor:
        try:
            cholesky_sigma = np.linalg.cholesky(sigma)
            has_non_positive_definite_minor = False
        except np.linalg.LinAlgError:
            for i in range(len(sigma)):
                sigma[i, i] += mcv
            mcv *= 10
    return np.dot(r,cholesky_sigma.T) + mu

cdef unsigned int numParametersGaussian(unsigned int n):
    return <unsigned int>(0.5 * n * (n + 3.0))

def stableMahalanobis(x, mu, sigma):
    """ Stable version of the Mahalanobis distance
    
    Parameters
    ----------
    x : input vector
    mu : mean of the input signal
    sigma : variance-covariance matrix of the input signal
    """
    has_nans = True
    mcv = 0.00001
    while has_nans:
        try:
            inv_sigma = np.array(np.linalg.inv(sigma), dtype = np.double)
            has_nans = False
        except np.linalg.LinAlgError:
            inv_sigma = sigma + np.eye(len(sigma), dtype = np.double) * mcv # TODO - ERROR
            try:
                inv_sigma = np.linalg.inv(inv_sigma)
            except:
                inv_sigma = np.nan_to_num(inv_sigma)
                inv_sigma = np.linalg.inv(inv_sigma)
            mcv *= 10 
    q = (cdist(x, mu[np.newaxis],"mahalanobis", VI = inv_sigma)**2).reshape(-1) # TODO : réécrire
    """
    delta = x - mu[np.newaxis]
    distance = np.sqrt(np.dot(np.dot(delta, inv_A), delta))
    """
    q[np.isnan(q)] = NUMPY_INF_VALUE
    return q

cdef cnp.ndarray GaussianLoglikelihood(cnp.ndarray obs, cnp.ndarray mu, 
                                       cnp.ndarray sigma, cnp.ndarray mv_indexes):
    """ Returns a matrix representing the log-likelihood of the distribution
    
    Parameters
    ----------
    obs : input signal
    mu : mean of the input signal
    sigma : variance-covariance matrix of the input signal
    mv_indexes : positions of the missing values in [[obs]]
    """
    cdef Py_ssize_t nobs = obs.shape[0]
    cdef Py_ssize_t ndim = obs.shape[1] 
    cdef Py_ssize_t nmix = len(mu)
    cdef double mcv
    cdef Py_ssize_t k, j
    cdef double dln2pi = ndim * libc.math.log(2.0 * libc.math.M_PI)
    lnf = np.empty((nobs, nmix))
    for k in range(nmix):
        try:
            det_sigma = np.linalg.det(sigma[k])
        except ValueError:
            sigma[k] = np.nan_to_num(sigma[k])
            det_sigma = np.linalg.det(sigma[k])
        lndetV = libc.math.log(det_sigma)
        mcv = 0.00001
        while det_sigma == 0 or np.isnan(lndetV):
            for j in range(ndim):
                sigma[k, j, j] += mcv
            mcv *= 10
            det_sigma = np.linalg.det(sigma[k])
            lndetV = libc.math.log(det_sigma)
        
        q = stableMahalanobis(obs, mu[k], sigma[k])
        # TODO : adjust lnf by taking into account self.missing_values_indexes 
        lnf[:, k] = -0.5 * (dln2pi + lndetV + q) # TODO : check the log-likelihood constraints
    return lnf

cdef logsum(matrix, axis = None):
    M = matrix.max(axis)
    if axis and matrix.ndim > 1:
        shape = list(matrix.shape)
        shape[axis] = 1
        M.shape = shape
    sum = np.log(np.sum(eexp(matrix - M), axis))
    sum += M.reshape(sum.shape)
    if axis:
        sum[np.isnan(sum)] = LOG_ZERO
    return sum

def normalizeMatrix(matrix, axis = None):
    matrix += np.finfo(float).eps
    sum = matrix.sum(axis)
    if axis and matrix.ndim > 1:
        sum[sum == 0] = 1
        shape = list(matrix.shape)
        shape[axis] = 1
        sum.shape = shape
    return matrix / sum



cdef class BaseHMM:
    """ Base class for all the library's HMM classes, which provides support
    for all the existing HMM architectures (ergodic, linear, cyclic, Bakis,...)
    and the most used probability distributions. It has been implemented mostly
    for machine learning purposes, this can be achieved by fitting several sets
    of observations, evaluate how the unlabeled data matches each of the trained
    models, and finally pick the less costly one.
    
    Example
    -------
    >>> hmm  = BaseHMM(15, distribution = DISTRIBUTION_LINEAR)
    >>> hmm.fit(dataset1, n_iterations = 25)
    >>> hmm2 = BaseHMM(15, distribution = DISTRIBUTION_LINEAR)
    >>> hmm2.fit(dataset2, n_iterations = 25)
    >>> best_fit = min(hmm.score(dataset3), hmm2.score(dataset3))
    
    In the example, the most probable label for the unlabeled data is the model
    among (hmm, hmm2) which minimizes its information criterion
    
    Parameters
    ----------
    n_states : number of hidden states of the model
    architecture : type of architecture of the HMM
                   [ARCHITECTURE_ERGODIC] All the nodes are connected in a reciprocal way.
                   [ARCHITECTURE_LINEAR] Each node is connected to the next node in a one_sided way.
                   [ARCHITECTURE_BAKIS] Each node is connected to the n next nodes in a one_sided way.
                   [ARCHITECTURE_LEFT_TO_RIGHT] Each node is connected to all the next nodes in a 
                                                one_sided way.
                   [ARCHITECTURE_CYCLIC] Each node is connected to the next one in a one_sided way,
                                         except the last node which loops back to the first one.
    distribution : type of the input signal's distribution
                   [DISTRIBUTION_GAUSSIAN] For variable following a normal law
                   [DISTRIBUTION_MULTINOMIAL] For discrete variables with a finite set of values
    missing_value : numeric value representing the missing values in the observations
                    If no value is provided, the training and scoring algorithms will not check
                    for the missing values before processing the data.
    
    Attributes
    ----------
    initial_probs : array containing the probability, for each hidden state, to be processed at first
    transition_probs : matrix where the element i_j represents the probability to move to the hidden
                       state j knowing that the current state is the state i  
    """

    cdef unsigned int architecture, distribution, use_implementation
    cdef Py_ssize_t n_states
    cdef cnp.ndarray initial_probs, transition_probs, previous_A
    cdef cnp.ndarray ln_initial_probs, ln_transition_probs
    cdef cnp.ndarray mu
    cdef cnp.ndarray previous_mu, sigma, previous_sigma, MU, SIGMA
    cdef unsigned int (*numParameters)(unsigned int)
    cdef cnp.ndarray (*loglikelihood)(cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray)
    # Arguments for handling missing values
    cdef double missing_value
    cdef cnp.ndarray mv_indexes
    # Arguments for the input-output HMM architecture
    cdef object pi_state_subnetwork, state_subnetworks, output_subnetworks
    cdef size_t n_classes

    def __cinit__(self, n_states, distribution = DISTRIBUTION_GAUSSIAN,
                  architecture = ARCHITECTURE_LINEAR, missing_value = DEFAULT_MISSING_VALUE,
                  use_implementation = USE_LIN_IMPLEMENTATION):
        if not (ARCHITECTURE_LINEAR <= architecture <= ARCHITECTURE_CYCLIC): 
            raise NotImplementedError("This architecture is not supported yet")
        self.use_implementation = use_implementation
        self.architecture = architecture
        self.distribution = distribution
        self.n_states = n_states
        self.missing_value = missing_value
        if distribution == DISTRIBUTION_GAUSSIAN:
            self.numParameters = &numParametersGaussian
            self.loglikelihood = &GaussianLoglikelihood
        else:
            raise NotImplementedError("This distribution is not supported yet") # TODO
        
    def getMu(self):
        return self.mu
    
    def getA(self):
        return self.transition_probs
            
    cdef int initParameters(self, obs) except -1:
        """ Initialize the parameters of the model according to the output of the
        parameter estimation algorithm. The latter (a clustering or a change point 
        detection algorithm) must be executed before this method is called.
        if architecture == ARCHITECTURE_ERGODIC : the parameter estimation algorithm
        must be a clustering algorithm.
        if architecture == ARCHITECTURE_LINEAR : the parameter estimation algorithm
        must be the change point detection algorithm.
        """
        cdef Py_ssize_t n = obs.shape[0]
        cdef Py_ssize_t n_dim = obs.shape[1]
        cdef size_t i
        self.setMissingValues(obs)
        
        if self.architecture == ARCHITECTURE_LINEAR:
            cpd = BatchCPD(n_keypoints = self.n_states, window_padding = 1,
                           cost_func = SUM_OF_SQUARES_COST, aprx_degree = 2)
            cpd.detectPoints(obs, self.MU, self.SIGMA)
            printf("\tParameter estimation finished\n")
            keypoint_indexes = cpd.getKeypoints()
            # self.n_states = len(keypoint_indexes)
            self.transition_probs = np.zeros((self.n_states, self.n_states), dtype = np.float)
            self.transition_probs[-1, -1] = 1.0
            for i in range(self.n_states - 1):
                a_ij = <float>1.0 / <float>(keypoint_indexes[i + 1] - keypoint_indexes[i])
                self.transition_probs[i, i + 1] = a_ij
                self.transition_probs[i, i] = 1.0 - a_ij
            self.initial_probs = np.zeros(self.n_states, dtype = np.float)
            self.initial_probs[0] = 1.0
            self.mu = np.empty((self.n_states, n_dim), dtype = np.double)
            self.sigma = np.empty((self.n_states, n_dim, n_dim), dtype = np.double)
            for i in range(self.n_states):
                segment = obs[keypoint_indexes[i]:keypoint_indexes[i + 1], :]
                self.mu[i] = segment.mean(axis = 0)
                self.sigma[i] = np.cov(segment.T)
            # TODO : problem : self.mu[-1] contains outliers
        elif self.architecture == ARCHITECTURE_ERGODIC:
            self.mu = kMeans(obs, self.n_states)
            self.sigma = np.tile(np.identity(obs.shape[1]),(self.n_states, 1, 1))
            self.initial_probs = np.tile(1.0 / self.n_states, self.n_states)
            self.transition_probs = dirichlet([1.0] * self.n_states, self.n_states)

        self.mu = np.nan_to_num(self.mu)
        self.sigma = np.nan_to_num(self.sigma)
        self.previous_mu = np.copy(self.mu)
        self.previous_sigma = np.copy(self.sigma)
        if self.use_implementation == USE_LOG_IMPLEMENTATION:
            self.ln_initial_probs = np.nan_to_num(elog(self.initial_probs))
            self.ln_transition_probs = np.nan_to_num(elog(self.transition_probs))
        
    cdef forwardProcedure(self, cnp.ndarray lnf, cnp.ndarray ln_alpha):
        """ Implementation of the forward procedure 
        (1st part of the forward-backward algorithm)
        
        Parameters
        ----------
        lnf : log-probability of the input signal's distribution
        ln_alpha : logarithm of the alpha matrix
                   See the documentation for further information about alpha
        Returns
        -------
        The loglikelihood of the forward procedure 
        """ 
        cdef Py_ssize_t t, T = len(lnf)
        with np.errstate(over = 'ignore'):
            ln_alpha[0, :] = np.nan_to_num(self.ln_initial_probs + lnf[0, :])
            for t in range(1, T):
                ln_alpha[t, :] = logsum(ln_alpha[t - 1, :] + self.ln_transition_probs.T, 1) + lnf[t, :]
            ln_alpha = np.nan_to_num(ln_alpha)
        return logsum(ln_alpha[-1, :])

    cdef backwardProcedure(self, lnf, ln_beta):
        """ Implementation of the backward procedure 
        (2nd part of the forward-backward algorithm)
        
        Parameters
        ----------
        lnf : log-probability of the input signal's distribution
        ln_alpha : logarithm of the beta matrix
                   See the documentation for further information about beta
        Returns
        -------
        The loglikelihood of the backward procedure
        """
        cdef Py_ssize_t t, T = len(lnf)
        with np.errstate(over = 'ignore'):
            ln_beta[T - 1, :] = 0.0
            for t in range(T - 2, -1, -1):
                ln_beta[t, :] = logsum(self.ln_transition_probs + lnf[t + 1, :] + ln_beta[t + 1, :], 1)
            ln_beta = np.nan_to_num(ln_beta)
        return logsum(ln_beta[0, :] + lnf[0, :] + self.ln_initial_probs)
    
    def E_step(self, lnf, ln_alpha, ln_beta, ln_eta):
        cdef Py_ssize_t i, j, t, T = len(lnf)
        lnP_f = np.nan_to_num(self.forwardProcedure(lnf, ln_alpha))
        lnP_b = np.nan_to_num(self.backwardProcedure(lnf, ln_beta))
        if abs((lnP_f - lnP_b) / lnP_f) > 1.0e-6:
            printf("Error. Forward and backward algorithm must produce the same loglikelihood.\n")
        with np.errstate(over = 'ignore'):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    for t in range(T - 1):
                        ln_eta[t, i, j] = ln_alpha[t, i] + self.ln_transition_probs[i, j,] + lnf[t+1,j] + ln_beta[t+1,j]      
                ln_eta -= lnP_f
                
        ln_eta = np.nan_to_num(ln_eta)
        ln_gamma = ln_alpha + ln_beta - lnP_f
        return ln_eta, ln_gamma, lnP_f

    def logFit(self, obs, mu, sigma, n_iterations = 100, 
            dynamic_features = False, delta_window = 1):
        """
        Launches the iterative Baum-Welch algorithm for parameter re-estimation.
        Call this method for training purposes, after having executed a parameter estimation
        algorithm such as a clustering function.
        
        Parameters
        ----------
        obs : input signal
        n_iterations : maximum number of iterations
        convergence_threshold : minimum decrease rate for the cost function
                                Under this value, we consider that the algorithm has converged.
                                If [[n_terations]] is small enough, the algorithm may stop 
                                before the convergence criterion can be fulfilled.
        dynamic_features : 
        sigma : variance-covariance matrix of the WHOLE signal
        """
        self.MU = mu
        self.SIGMA = sigma
        assert(len(obs.shape) == 2)
        if dynamic_features:
            deltas, delta_deltas = self.getDeltas(obs, delta_window = delta_window)
            obs = np.concatenate((obs, deltas, delta_deltas), axis = 1)
            
        T, D = obs.shape[0], obs.shape[1]
        self.initParameters(obs)
        ln_alpha = np.zeros((T, self.n_states))
        ln_beta = np.zeros((T, self.n_states))
        ln_eta = np.zeros((T - 1, self.n_states, self.n_states))

        cdef long convergence_threshold = <long>0.0001
        cdef bint has_converged = False
        old_F = 1.0e20
        i = 0
        while i < n_iterations and not has_converged:
            lnf = GaussianLoglikelihood(obs, self.mu, self.sigma, self.mv_indexes)
            ln_eta, ln_gamma, lnP = self.E_step(lnf, ln_alpha, ln_beta, ln_eta)
            F = - lnP
            dF = F - old_F
            if(np.abs(dF) < convergence_threshold):
                has_converged = True
            old_F = F
            
            gamma = ieexp2d(ln_gamma) # inplace exp function
            for k in range(self.n_states):
                post = gamma[:, k]
                post_sum = post.sum()
                norm = 1.0 / post_sum if post_sum != 0.0 else -LOG_ZERO
                temp = np.nan_to_num(np.dot(post * obs.T, obs))
                avg_sigma = temp * norm
                self.mu[k] = np.nan_to_num(np.dot(post, obs) * norm)
                self.sigma[k] = np.nan_to_num(avg_sigma - np.outer(self.mu[k], self.mu[k]))
                
                if np.all(self.mu[k] == 0):
                    self.mu[k] = self.previous_mu[k]
                
            is_nan = (self.mu == np.nan)
            self.mu[is_nan] = self.previous_mu[is_nan]
            is_nan = (self.sigma == np.nan)
            self.sigma[is_nan] = self.previous_sigma[is_nan]
            self.previous_mu[:] = self.mu[:]
            self.previous_sigma[:] = self.sigma[:]
            i += 1
            
        self.transition_probs = eexp(self.ln_transition_probs)
        eigenvalues = np.linalg.eig(self.transition_probs.T)
        self.initial_probs = normalizeMatrix(np.abs(eigenvalues[1][:, eigenvalues[0].argmax()]))
        
    cdef void linForwardProcedure(self, cnp.ndarray alpha, cnp.ndarray pi, cnp.ndarray A, cnp.ndarray B):
        cdef Py_ssize_t i, j
        cdef Py_ssize_t T = alpha.shape[0]
        cdef Py_ssize_t n = alpha.shape[1]
        cdef double transition_seq_prob
        for i in range(n):
            alpha[0, i] = pi[i] * B[0, i]
        for t in range(1, T):
            for j in range(n):
                transition_seq_prob = 0
                for i in range(n):
                    transition_seq_prob += alpha[t - 1, i] * A[i, j]
                alpha[t, j] = transition_seq_prob * B[t, j]
            
    cdef void linBackwardProcedure(self, cnp.ndarray beta, cnp.ndarray A, cnp.ndarray B):
        cdef Py_ssize_t i, j
        cdef Py_ssize_t T = beta.shape[0]
        cdef Py_ssize_t n = beta.shape[1]
        cdef double temp
        for i in range(n):
            beta[T - 1, i] = 1
        for t in range(0, T - 1):
            for i in range(n):
                temp = 0
                for j in range(n):
                    temp += beta[t + 1, j] * A[i, j] * B[t + 1, j]
                beta[t, i] = temp
        
    def linFit(self, obs, mu, sigma, n_iterations = 100, 
            dynamic_features = False, delta_window = 1):
        self.MU = mu
        self.SIGMA = sigma
        assert(len(obs.shape) == 2)
        if dynamic_features:
            deltas, delta_deltas = self.getDeltas(obs, delta_window = delta_window)
            obs = np.concatenate((obs, deltas, delta_deltas), axis = 1)
        self.initParameters(obs)
        cdef bint has_converged = False
        cdef Py_ssize_t i, j, t, k
        cdef Py_ssize_t T = obs.shape[0]
        cdef Py_ssize_t n = obs.shape[1]
        cdef cnp.ndarray alpha = np.empty((T, n), dtype = np.double)
        cdef cnp.ndarray beta = np.empty((T, n), dtype = np.double)
        cdef cnp.ndarray gamma = np.empty((T, n), dtype = np.double)
        cdef cnp.ndarray eta = np.empty((T - 1, n, n), dtype = np.double)
        cdef cnp.ndarray pi = self.initial_probs
        cdef cnp.ndarray A = self.transition_probs
        self.previous_A = np.copy(A)
        cdef cnp.ndarray B
        cdef double bayes_denom, eta_sum, gamma_sum
        cdef cnp.ndarray bayes_num = np.empty((n, n), dtype = np.double)
        B = GaussianLoglikelihood(obs,self.mu,self.sigma, self.mv_indexes)
        k = 0
        while k < n_iterations and not has_converged:
            self.linForwardProcedure(alpha, pi, A, B)
            self.linBackwardProcedure(beta, A, B)
            
            for t in range(T):
                bayes_denom = 0
                for j in range(n):
                    bayes_denom += alpha[t, j] * beta[t, j]
                for i in range(n):
                    gamma[t, i] = (alpha[t, i] * beta[t, i]) / bayes_denom
            gamma = np.nan_to_num(gamma)

            for t in range(0, T - 1):
                bayes_denom = 0
                for i in range(n):
                    for j in range(n):
                        bayes_num[i, j] = alpha[t, i] * A[i, j] * beta[t + 1, j] * B[t + 1, j]
                        bayes_denom += bayes_num[i, j]
                for i in range(n):
                    for j in range(n):
                        eta[t, i, j] = bayes_num[i, j] / bayes_denom
            eta = np.nan_to_num(eta)

            for i in range(n):
                pi[i] = gamma[0, i]
                for j in range(n):
                    eta_sum = gamma_sum = 0
                    for t in range(0, T - 1):
                        eta_sum += eta[t, i, j]
                        gamma_sum += gamma[t, i]
                    A[i, j] = eta_sum / gamma_sum if gamma_sum > 0 else self.previous_A[i, j]
            pi = np.nan_to_num(pi)
            A = np.nan_to_num(A)

            for i in range(n):
                post = gamma[:, i]
                post_sum = post.sum()
                norm = 1.0 / post_sum if post_sum != 0.0 else -LOG_ZERO
                temp = np.nan_to_num(np.dot(post * obs.T, obs))
                avg_sigma = temp * norm
                self.mu[i] = np.nan_to_num(np.dot(post, obs) * norm)
                self.sigma[i] = np.nan_to_num(avg_sigma - np.outer(self.mu[i], self.mu[i]))
                if np.all(self.mu[i] == 0):
                    self.mu[i] = self.previous_mu[i]

            is_nan = (self.mu == np.nan)
            self.mu[is_nan] = self.previous_mu[is_nan]
            is_nan = (self.sigma == np.nan)
            self.sigma[is_nan] = self.previous_sigma[is_nan]
            self.previous_mu[:] = self.mu[:]
            self.previous_sigma[:] = self.sigma[:]
            k += 1

        for i in range(n):
            A[i, :] = A[i, :] / A[i, :].sum()
        self.transition_probs = A
        self.initial_probs = pi / pi.sum()

    def fit(self, obs, mu, sigma, **kwargs):
        if self.use_implementation == USE_LIN_IMPLEMENTATION:
            self.linFit(obs, mu, sigma, **kwargs)
        else:
            self.logFit(obs, mu, sigma, **kwargs)
        
    def fitIO(self, inputs, targets = None, mu = None, sigma = None, n_iterations = 5, n_epochs = 1,
              dynamic_features = False, delta_window = 1, n_classes = 2):
        self.MU = mu
        self.SIGMA = sigma
        self.n_classes = n_classes
        cdef Py_ssize_t i, j, k, l, p
        cdef size_t n_sequences = len(inputs)
        assert(n_sequences == len(targets))
        cdef cnp.ndarray T = np.empty(n_sequences)
        for p in range(n_sequences):
            T[p] = len(inputs[p])
        cdef cnp.ndarray U = typedListTo3DPaddedTensor(inputs, T)
        cdef size_t m = U[0].shape[1]
        cdef size_t n = self.n_states
        cdef size_t r = n_classes
        cdef size_t n_hidden = 16 # Number of hidden units must be determined with the validation set
        cdef object N = newStateSubnetworks(n, m, n_hidden, n, network_type = SUBNETWORK_STATE)
        self.state_subnetworks = N
        cdef object O = newStateSubnetworks(n, m, n_hidden, r, network_type = SUBNETWORK_OUTPUT)
        self.output_subnetworks = O
        piN = newStateSubnetworks(1, m, n_hidden, n, network_type = SUBNETWORK_PI_STATE)[0]
        self.pi_state_subnetwork = piN
        cdef cnp.ndarray loglikelihood = np.zeros(n_iterations)
        cdef object alpha = new3DVLMArray(n_sequences, n, T, dtype = np.float)
        cdef object beta = new3DVLMArray(n_sequences, n, T, dtype = np.float)
        cdef object gamma = new3DVLMArray(n_sequences, n, T, dtype = np.float)
        cdef object xi = new3DVLMArray(n_sequences, n, T, n, dtype = np.double)
        cdef cnp.ndarray A = np.empty((n, n), dtype = np.float)
        cdef cnp.ndarray initial_probs = np.empty((n), dtype = np.float)
        cdef cnp.ndarray memory = np.empty((n_sequences, n), dtype = np.float)
        cdef cnp.ndarray new_internal_state = np.empty(n, dtype = np.float)
        cdef object B = new3DVLMArray(n_sequences, n, T, r, dtype = np.double)
        for iter in range(n_iterations):
            print("Iteration number %i..." % iter)
            """ Forward procedure """
            for j in range(n_sequences):
                for i in range(n):
                    for k in range(T[j]):
                        B[j, i, k, :] = O[i].processOutput(U[j][k, :]).eval()
            printf("\tDensities computed\n")
            for j in range(n_sequences):
                initial_probs = piN.processOutput(U[j][0, :]).eval() # Processing sequence j for all times t
                memory[j, :] = initial_probs
                sequence_probs = np.multiply(B[j, :, 0, targets[j][0]], initial_probs)
                loglikelihood[iter] = np.log(np.sum(sequence_probs))
                for k in range(1, T[j]):
                    new_internal_state[:] = 0
                    for i in range(n):
                        A[i] = N[i].processOutput(U[j][k, :]).eval()
                        new_internal_state[:] += memory[j, i] * A[i]
                    memory[j, :] = new_internal_state
                    for i in range(n):
                        alpha[j, i, k] = 0
                        for l in range(n):
                            alpha[j, i, k] += alpha[j, l, k - 1] * A[l, i]
                        alpha[j, i, k] *= B[j, i, k, targets[j][k]]
                    print(j, k)
                    loglikelihood[iter] += np.log(np.sum(sequence_probs))
            printf("\tAlpha probabilities computed\n")
            """ Backward procedure """
            for j in range(n_sequences):
                beta[j, :, -1] = 1
                for k in range(T[j] - 2, -1, -1):
                    for i in range(n):
                        # TODO : already computed -> spare some computation
                        A[i, :] = N[i].processOutput(U[j][k + 1, :]).eval()
                    # beta[j][:, k] = np.dot(A, np.multiply(B[j][:, k + 1], beta[j][:, k + 1]))
                    for i in range(n):
                        beta[j, i, k] = 0
                        for l in range(n): # TODO : SUPER SLOW
                            beta[j, i, k] += beta[j, l, k + 1] * A[i, l] * B[j, l, k + 1, targets[j][k + 1]]
                    print(j, k)
            printf("\tBeta probabilities computed\n")
            """ Forward-Backward xi computation """
            for j in range(n_sequences):
                gamma[j] = np.multiply(alpha[j], beta[j]) # TO REMOVE IF USELESS
            for j in range(n_sequences):
                denominator = np.sum(alpha[j, :, -1])
                xi[j, :, 0, :] = 0 # TODO ???
                for k in range(T[j] - 1):
                    for i in range(n):
                        A[i, :] = N[i].processOutput(U[j][k + 1, :]).eval()
                        for l in range(n):
                            # xi[j, :, k, :] = np.multiply(A, alpha[j, :, k] * np.multiply(B[j, :, k + 1, targets[j][k + 1]], beta[j, :, k + 1]))
                            xi[j, i, k + 1, l] = beta[j, i, k + 1] * alpha[j, l, k] * A[i, l] / denominator
            printf("\tXi tensor computed\n")
            """ E-Step """
            """
            state_frequencies = np.zeros(n, dtype = np.float)
            for j in range(n_sequences):
                state_frequencies += np.sum(gamma[j], axis = 1)
            obs_vector = np.zeros((n, r), dtype = np.int)
            for j in range(n_sequences):
                for k in range(r):
                    obs_vector[:, k] += np.sum(gamma[j, :, np.where(targets[j][-1] == k)[0]], axis = 1) # TODO : check np.where indices
            """
            printf("\tEnd of expectation step\n")
            """ M-Step """
            # TODO : Minimize function for the piW network
            n_epochs
            piN.train(U, gamma, n_epochs = n_epochs)
            for j in range(n):
                N[j].train(U, xi, n_epochs = n_epochs)
                O[j].train(U, memory, n_epochs = n_epochs)
            printf("\tEnd of maximization step\n")
                
    def predictIO(self, input):
        """ Viterbi decoder for input-output HMM """
        assert(len(input.shape) == 2)
        cdef Py_ssize_t T = input.shape[0]
        cdef object piN = self.pi_state_subnetwork
        cdef object N   = self.state_subnetworks
        cdef object O   = self.output_subnetworks
        cdef cnp.ndarray state_sequence = np.empty((self.n_classes, T), dtype = np.int8)
        cdef cnp.ndarray output_sequence = np.empty((self.n_classes, T), dtype = np.float16)
        cdef cnp.ndarray current_eta = np.empty((self.n_states, self.n_classes), dtype = np.float16)
        cdef cnp.ndarray memory = np.zeros((self.n_classes, self.n_states), dtype = np.double)
        cdef Py_ssize_t i, t
        
        memory = np.tile(np.log2(piN.processOutput(input[0]).eval()), (self.n_classes, 1))
        for i in range(self.n_states):
            current_eta[i, :] = O[i].processOutput(input[0]).eval()
        for i in range(self.n_classes):
            memory[i] += np.log2(current_eta[:, i])
            state_sequence[i, 0] = memory[i].argmax()
        for t in range(1, T):
            for i in range(self.n_classes):
                a = N[state_sequence[i, t - 1]].processOutput(input[t]).eval().reshape(self.n_states)
                memory[i, :] += np.log2(a)
            for i in range(self.n_states):
                current_eta[i, :] = O[i].processOutput(input[t]).eval()
            for i in range(self.n_classes):
                memory[i, :] += np.log2(current_eta[:, i])
                state_sequence[i, t] = memory[i].argmax()
            print(current_eta[state_sequence[0, t], 0], current_eta[state_sequence[1, t], 0], state_sequence[0, t], state_sequence[1, t])
        print(state_sequence)
        return np.argmax(memory.max(axis = 1))

    def noisedDistribution(self, state):
        return GaussianGenerator(self.mu[state], self.sigma[state])
    
    def cleanDistribution(self, state):
        return self.mu[state]
    
    cdef getDeltas(self, obs, delta_window = 1):
        cdef Py_ssize_t n = len(obs)
        cdef cnp.double_t[:] deltas = np.zeros(obs.shape)
        cdef cnp.double_t[:] delta_deltas = np.zeros(obs.shape)
        cdef Py_ssize_t k, theta
        cdef double num, den
        for k in range(delta_window, n - delta_window):
            num = den = 0.0
            for theta in range(1, delta_window + 1):
                num += theta * (obs[k + theta] - obs[k - theta])
                den += theta ** 2
            den = 2.0 * den
            deltas[k] = num / den
        for k in range(2 * delta_window, n - 2 * delta_window):
            num = den = 0.0
            for theta in range(1, delta_window + 1):
                num += theta * (deltas[k + theta] - deltas[k - theta])
                den += theta ** 2
            den = 2.0 * den
            delta_deltas[k] = num / den
        return deltas, delta_deltas
    
    def randomSequence(self, T, start_from_left = True, dynamic_features = False, noised_distribution = False):
        distribution_func = self.noisedDistribution if noised_distribution else self.cleanDistribution        
        N, D = self.mu.shape[0], self.mu.shape[1]
        pi_cdf = self.initial_probs.cumsum()
        A_cdf = self.transition_probs.cumsum(1)
        states = np.zeros(T, dtype = np.int)
        observations = np.zeros((T, D))
        r = random(T)
        if not start_from_left:
            states[0] = (pi_cdf > r[0]).argmax()
        else:
            states[0] = 0
        observations[0] = distribution_func(states[0])
        for t in range(1,T):
            states[t] = (A_cdf[states[t-1]] > r[t]).argmax()
            observations[t] = distribution_func(states[t])
        return states, observations
    
    cdef void setMissingValues(self, cnp.ndarray observations):
        if self.missing_value == DEFAULT_MISSING_VALUE:
            self.mv_indexes = getMissingValuesIndexes(observations, self.missing_value)
        else:
            self.mv_indexes = np.empty(0)

    def score(self, obs, mode = CRITERION_AICC):
        """
        Evaluates how the model fits the data, by taking into account the complexity
        (number of parameters) of the model. The best model both minimizes its complexity
        and maximizes its likelihood
        
        Parameters
        ----------
        obs : input signal
        mode : score function to use
               [CRITERION_AIC] Akaike Information Criterion
               [CRITERION_AICC] Akaike Information Criterion with correction 
                                (for small-sample sized models)
               [CRITERION_BIC] Bayesian Information Criterion
               [CRITERION_LIKELIHOOD] Negative Likelihood
        """
        self.setMissingValues(obs)
        n = obs.shape[0]        
        cdef cnp.ndarray lnf = GaussianLoglikelihood(obs,self.mu,self.sigma, self.mv_indexes)
        cdef size_t T = len(obs)
        cdef cnp.ndarray ln_alpha = np.zeros((T, self.n_states)) # TODO : replace np.zeros by np.empty
        cdef cnp.ndarray ln_beta = np.zeros((T, self.n_states))
        cdef cnp.ndarray ln_eta = np.zeros((T - 1, self.n_states, self.n_states))    
        ln_eta, ln_gamma, lnP = self.E_step(lnf, ln_alpha, ln_beta, ln_eta)
        nmix, ndim = self.mu.shape[0], self.mu.shape[1]
        # k == Complexity of the model
        k = self.n_states * (1.0 + self.n_states) + nmix * self.numParameters(ndim)
        if mode == CRITERION_AIC:
            criterion = 2 * k - 2 * lnP
        elif mode == CRITERION_AICC:
            criterion = 2 * k - 2 * lnP + float(2 * k * (k + 1)) / float(n - k - 1)
        elif mode == CRITERION_BIC:
            criterion = k * elog(n) - lnP
        elif mode == CRITERION_LIKELIHOOD:
            criterion = - lnP
        elif mode == CRITERION_NEG_LIKELIHOOD:
            criterion = lnP
        else:
            raise NotImplementedError("The given information criterion is not supported")
        return criterion
    
    def decode(self, obs):
        """ Returns the index of the most probable state, given the observations """
        cdef cnp.ndarray lnf = GaussianLoglikelihood(obs, self.mu, self.sigma, self.mv_indexes)
        cdef size_t T = len(obs)
        cdef cnp.ndarray ln_alpha = np.zeros((T, self.n_states)) # TODO : replace np.zeros by np.empty
        cdef cnp.ndarray ln_beta = np.zeros((T, self.n_states))
        cdef cnp.ndarray ln_eta = np.zeros((T - 1, self.n_states, self.n_states))       
        ln_eta, ln_gamma, lnP = self.E_step(lnf, ln_alpha, ln_beta, ln_eta)
        gamma = ieexp2d(ln_gamma)
        return gamma.argmax(1)
    
    cpdef cSave(self, char* filename):
        cdef size_t i, j, k
        cdef Py_ssize_t n_dim = self.sigma.shape[1]
        cdef FILE* ptr_fw = fopen(filename, "wb")
        if ptr_fw == NULL:
            printf("Error. Could not open file %s\n", filename)
            exit(EXIT_FAILURE)
        cdef cnp.double_t* mu = <cnp.double_t*>self.mu.data
        cdef cnp.double_t* sigma = <cnp.double_t*>self.sigma.data
        cdef cnp.float_t* initial_probs = <cnp.float_t*>self.initial_probs.data
        cdef cnp.float_t* transition_probs = <cnp.float_t*>self.transition_probs.data
        fwrite(&(self.architecture), sizeof(unsigned int), sizeof(unsigned int), ptr_fw)
        fwrite(&(self.distribution), sizeof(unsigned int), sizeof(unsigned int), ptr_fw)
        fwrite(&(self.n_states), sizeof(Py_ssize_t), sizeof(Py_ssize_t), ptr_fw)
        fwrite(&n_dim, sizeof(Py_ssize_t), sizeof(Py_ssize_t), ptr_fw)
        fwrite(&mu, sizeof(cnp.double_t), self.n_states * n_dim * sizeof(cnp.double_t), ptr_fw)
        fwrite(&sigma, sizeof(cnp.double_t), self.n_states * n_dim * n_dim * sizeof(cnp.double_t), ptr_fw)
        fwrite(&initial_probs, sizeof(cnp.float_t), self.n_states * sizeof(cnp.float_t), ptr_fw)
        fwrite(&transition_probs, sizeof(cnp.float_t), self.n_states * self.n_states * sizeof(cnp.float_t), ptr_fw)
        fclose(ptr_fw)
        
    cpdef cLoad(self, char* filename):
        cdef Py_ssize_t i, j, k
        cdef Py_ssize_t n_dim
        cdef FILE* ptr_fr = fopen(filename, "rb")
        if ptr_fr == NULL:
            printf("Error. Could not open file %s\n", filename)
            exit(EXIT_FAILURE)
        fread(&(self.architecture), sizeof(unsigned int), sizeof(unsigned int), ptr_fr)
        fread(&(self.distribution), sizeof(unsigned int), sizeof(unsigned int), ptr_fr)
        fread(&(self.n_states), sizeof(Py_ssize_t), sizeof(Py_ssize_t), ptr_fr)
        fread(&n_dim, sizeof(Py_ssize_t), sizeof(Py_ssize_t), ptr_fr)
        self.mu = np.empty((self.n_states, n_dim), dtype = np.double)
        self.mu = np.empty((self.n_states, n_dim), dtype = np.double)
        cdef cnp.double_t* mu = <cnp.double_t*>self.mu.data
        cdef double* sigma = <double*>self.sigma.data
        cdef float* initial_probs = <float*>self.initial_probs.data
        cdef float* transition_probs = <float*>self.transition_probs.data
        fclose(ptr_fr)
        
    def pySave(self, filename):
        attributes = {
            "architecture" : int(self.architecture),
            "distribution" : int(self.distribution),
            "n_states" : int(self.n_states),
            "initial_probs" : self.initial_probs,
            "transition_probs" : self.transition_probs,
            "mu" : self.mu, "sigma" : self.sigma,
            "MU" : self.mu, "SIGMA" : self.sigma,
            "missing_value" : self.missing_value,
            "piN" : self.pi_state_subnetwork,
            "N" : self.state_subnetworks,
            "O" : self.output_subnetworks
        }
        pickle.dump(attributes, open(filename, "wb"))
        
    def pyLoad(self, filename):
        attributes = pickle.load(open(filename, "rb"))
        self.architecture = attributes["architecture"]
        self.distribution = attributes["distribution"]
        self.n_states = attributes["n_states"]
        self.initial_probs = attributes["initial_probs"]
        self.transition_probs = attributes["transition_probs"]
        self.ln_initial_probs = np.nan_to_num(elog(self.initial_probs))
        self.ln_transition_probs = np.nan_to_num(elog(self.transition_probs))
        self.mu = attributes["mu"]
        self.sigma = attributes["sigma"]
        self.MU = self.mu = attributes["MU"]
        self.SIGMA = self.sigma = attributes["SIGMA"]
        self.missing_value = attributes["missing_value"]
        self.pi_state_subnetwork = attributes["piN"]
        self.state_subnetworks = attributes["N"]
        self.output_subnetworks = attributes["O"]
        
        

