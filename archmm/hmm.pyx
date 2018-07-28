# -*- coding: utf-8 -*-
# hmm.pyx : Base class for Hidden Markov Model
# author: Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math
from cython.parallel import parallel, prange

import copy
from abc import abstractmethod
import scipy.linalg
import scipy.cluster

from archmm.estimation.cpd import *
from archmm.estimation.cpd cimport *
from archmm.estimation.clustering import k_means
from archmm.stats import np_data_t, gaussian_log_proba
from archmm.utils import abstractpythonmethod, Architecture

cdef data_t INF = <data_t>np.inf
cdef data_t LOG_ZERO = -INF
cdef data_t ZERO = <data_t>0.0


cdef inline data_t _max(data_t[:] vec) nogil:
    cdef int i
    cdef data_t best_val = -INF
    for i in range(vec.shape[0]):
        if vec[i] > best_val:
            best_val = vec[i]
    return best_val


cdef inline data_t elogsum(data_t[:] vec) nogil:
    cdef int i
    cdef data_t s = 0.0
    cdef data_t offset = _max(vec)
    if libc.math.isinf(offset):
        return -INF
    for i in range(vec.shape[0]):
        s += libc.math.exp(vec[i] - offset)
    return libc.math.log(s) + offset


cdef inline void eexp2d(data_t[:, :] dest, data_t[:, :] src) nogil:
    cdef int i, j
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dest[i, j] = libc.math.exp(src[i, j]) \
                if src[i, j] != LOG_ZERO else 0.0


cdef inline void eexp3d(data_t[:, :, :] dest, data_t[:, :, :] src) nogil:
    cdef int i, j, k
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for k in range(src.shape[2]):
                dest[i, j, k] = libc.math.exp(src[i, j, k]) \
                    if src[i, j, k] != LOG_ZERO else 0.0


cdef inline void elog1d(data_t[:] dest, data_t[:] src) nogil:
    cdef int i
    for i in range(src.shape[0]):
        dest[i] = libc.math.log(src[i]) \
            if src[i] != 0 else LOG_ZERO


cdef inline void elog2d(data_t[:, :] dest, data_t[:, :] src) nogil:
    cdef int i, j
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dest[i, j] = libc.math.log(
                src[i, j]) if src[i, j] != 0 else LOG_ZERO


def create_buffer_list(X, shape, dtype):
    buffer_list = list()
    for sequence in X:
        buffer_shape = tuple([len(sequence)] + list(shape))
        buffer_list.append(np.empty(buffer_shape, dtype=dtype))
    return buffer_list


cdef class HMM:
    """ Base class for Hidden Markov Models

    Args:
        n_states (int):
            Number of hidden states
        arch (str):
            Topology name: can be either 'ergodic', 'linear' or 'cyclic'
    
    Attributes:
        n_features (int):
            Dimensionality of the observed samples
        ln_initial_probs (:obj:`np.ndarray`):
            Logarithm of state start probabilities, where ln_initial_probs[i]
            is the probability that the model starts in state i when generating
            a new sequence
        ln_transition_probs (:obj:`np.ndarray`):
            Logarithm of state transition probabilities, where ln_transition_probs[i, j]
            is the probability that the model switches from state i to state j
        initial_probs (:obj:`np.ndarray`):
            State start probabilities
        transition_probs (:obj:`np.ndarray`):
            State transition probabilities

    References:
        HMM by Dr Philip Jackson
        Centre for Vision Speech & Signal Processing,
        University of Surrey, Guildford GU2 7XH.
        .. http://homepages.inf.ed.ac.uk/rbf/IAPR/researchers/D2PAGES/TUTORIALS/hmm_isspr.pdf
    """

    def __init__(self, n_states, arch='ergodic'):
        self.n_states = n_states
        self.n_features = -1

        self.ln_initial_probs = np.empty(
            self.n_states, dtype=np_data_t)
        self.ln_transition_probs = np.empty(
            (self.n_states, self.n_states), dtype=np_data_t)
        self.initial_probs = np.copy(self.ln_initial_probs)
        self.transition_probs = np.copy(self.ln_transition_probs)

        arch = arch.lower().strip()
        if isinstance(arch, Architecture):
            self.transition_mask = arch.to_mask()
            self.arch = 'custom'
        else:
            self.transition_mask = np.asarray(
                self.init_topology(self.n_states, arch), dtype=np.int)
            self.arch = arch
    
    def init_topology(self, n_states, arch):
        topology = Architecture(n_states)
        if arch == 'ergodic':
            topology.add_edges_everywhere()
        elif arch == 'linear':
            topology.add_self_loops()
            topology.add_edges(lambda i: i+1)
        elif archmm == 'cyclic':
            topology.add_self_loops()
            topology.add_edges(lambda i: i+1)
            topology.add_edge(-1, 0)
        else:
            pass # TODO: raise exception
        return topology.to_mask()

    @abstractpythonmethod
    def get_num_params_per_state(self):
        pass
    
    @abstractpythonmethod
    def estimate_params(self, X):
        pass
    
    @abstractpythonmethod
    def emission_log_proba(self, X):
        pass
    
    @abstractpythonmethod
    def update_emission_params(self, X, gamma_s):
        pass
    
    cdef data_t[:] sample_one_from_state(self, int state_id) nogil:
        pass
    
    cdef data_t[:, :] compute_ln_phi(self, int sequence_id, int t) nogil:
        """ Compute the logarithm of state transition probabilities.
        Because of the independence and stationary assumptions,
        transition probabilities are constant. Therefore, this method always
        returns the same memory view.

        Args:
            sequence_id (int):
                Sequence index
            t (int):
                Current time (0 < t < T[p], where T[p] is the sequence length)
        
        Returns:
            :obj:`np.ndarray`:
                Logarithm of state transition probabilities
        """
        return self.ln_transition_probs

    cdef data_t forward(self,
                        data_t[:, :] lnf,
                        data_t[:, :] ln_alpha,
                        data_t[:] tmp,
                        int sequence_id) nogil:
        """ Computes forward probabilities.

        Args:   
            lnf (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing 
                the logarithm of emission probabilities for a given sequence
            ln_alpha (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing
                the logarithm of forward probabilities for a given sequence
            tmp (:obj:`np.ndarray`):
                Memory view for storing temporary values
            sequence_id (int):
                Sequence index

        Returns:
            float:
                Log-likelihood of the data, computed forward
        """
        cdef int i, j, t
        cdef int n_samples = lnf.shape[0]
        cdef data_t[:, :] ln_phi
        for i in range(self.n_states):
            ln_alpha[0, i] = self.ln_initial_probs[i] + lnf[0, i]
        for t in range(1, n_samples):
            ln_phi = self.compute_ln_phi(sequence_id, t)
            for j in range(self.n_states):
                for i in range(self.n_states):
                    tmp[i] = ln_alpha[t-1, i] + ln_phi[i, j]
                ln_alpha[t, j] = elogsum(tmp) + lnf[t, j]
        return elogsum(ln_alpha[n_samples-1, :])

    cdef data_t backward(self,
                         data_t[:, :] lnf,
                         data_t[:, :] ln_beta,
                         data_t[:] tmp,
                         int sequence_id):
        """ Computes backward probabilities.

        Args:   
            lnf (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing 
                the logarithm of emission probabilities for a given sequence
            ln_beta (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing
                the logarithm of backward probabilities for a given sequence
            tmp (:obj:`np.ndarray`):
                Memory view for storing temporary values
            sequence_id (int):
                Sequence index
        
        Returns:
            float:
                Log-likelihood of the data, computed backward
        """
        cdef Py_ssize_t i, j, t
        cdef int n_samples = lnf.shape[0]
        cdef data_t[:, :] ln_phi
        with nogil:
            for i in range(self.n_states):
                ln_beta[n_samples-1, i] = 0.0
            for t in range(n_samples-2, -1, -1):
                ln_phi = self.compute_ln_phi(sequence_id, t)
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        tmp[j] = ln_phi[i, j] + ln_beta[t+1, j] + lnf[t+1, j]
                    ln_beta[t, i] = elogsum(tmp)
        lnP_b = elogsum(np.asarray(ln_beta[0, :]) + np.asarray(lnf[0, :]) \
            + np.asarray(self.ln_initial_probs))
        return lnP_b

    cdef e_step(self,
                data_t[:, :] lnf,
                data_t[:, :] ln_alpha,
                data_t[:, :] ln_beta,
                data_t[:, :] ln_gamma,
                data_t[:, :, :] ln_xi,
                int sequence_id):
        """ Expectation step of the Expectation-Maximization step

        Args:   
            lnf (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing 
                the logarithm of emission probabilities for a given sequence
            ln_alpha (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing
                the logarithm of forward probabilities for a given sequence
            ln_beta (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing
                the logarithm of backward probabilities for a given sequence
            ln_gamma (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states) representing
                the logarithm of gamma probabilities for a given sequence
            ln_xi (:obj:`np.ndarray`):
                Memory view of shape (n_samples, n_states, n_states) representing
                the logarithm of xi probabilities for a given sequence
            tmp (:obj:`np.ndarray`):
                Memory view for storing temporary values
            sequence_id (int):
                Sequence index
        
        Returns:
            float:
                Log-likelihood of the data, computed forward
        """
        cdef Py_ssize_t i, j, t, k, l
        cdef int n_samples = ln_alpha.shape[0]
        cdef data_t[:, :] ln_phi

        cdef data_t[:] tmp = np.empty((self.n_states), dtype=np_data_t)
        cdef double lnP_f = self.forward(lnf, ln_alpha, tmp, sequence_id)
        cdef double lnP_b = self.backward(lnf, ln_beta, tmp, sequence_id)
        # TODO: CHECK THAT lnP_f AND lnP_b ARE ALMOST EQUAL

        with nogil:
            for i in range(self.n_states):
                for j in range(self.n_states):
                    ln_xi[0, i, j] = self.ln_initial_probs[i] + \
                        lnf[0, j] + ln_beta[0, j] - lnP_f # Can be replaced by any value
            for t in range(1, n_samples):
                ln_phi = self.compute_ln_phi(sequence_id, t)
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        ln_xi[t, i, j] = ln_alpha[t-1, i] + ln_phi[i, j] + \
                            lnf[t, j] + ln_beta[t, j] - lnP_f
        
            for t in range(n_samples):
                for i in range(self.n_states):
                    ln_gamma[t, i] = ln_alpha[t, i] + ln_beta[t, i] - lnP_f
        return lnP_f

    def baum_welch(self, X_s, max_n_iter=100, eps=1e-04):
        n_sequences = len(X_s)
        ln_alpha_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        ln_beta_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        ln_gamma_s = create_buffer_list(X_s, (self.n_states,), np_data_t)
        ln_xi_s = create_buffer_list(X_s, (self.n_states, self.n_states), np_data_t)
        lnf_s = [None for i in range(n_sequences)]
        lnP_s = np.empty(n_sequences) # Log-likelihood of each sequence

        cdef data_t[:] den, num

        old_F = 1.0e20
        for i in range(max_n_iter):
            print("\tIteration %i" % i)

            # Apply E-step on each sequence individually
            for p in range(n_sequences):
                lnf_s[p] = self.emission_log_proba(X_s[p])
                lnP_s[p] = self.e_step(lnf_s[p], ln_alpha_s[p], ln_beta_s[p],
                    ln_gamma_s[p], ln_xi_s[p], p)
            
            # Check log-likelihood of the data
            lnP = np.sum(lnP_s)
            F = -lnP
            dF = F - old_F
            if(np.abs(dF) < <long>eps):
                break
            old_F = F

            # Compute posteriors
            gamma_s = copy.deepcopy(ln_gamma_s)
            for j in range(n_sequences):
                eexp2d(gamma_s[j], ln_gamma_s[j])

            # Compute state start probabilities            
            self.initial_probs[:] = 0
            for p in range(n_sequences):
                self.initial_probs += gamma_s[p][0, :]
            self.initial_probs /= np.sum(self.initial_probs)
            elog1d(self.ln_initial_probs, self.initial_probs)

            # Compute state transition probabilities
            for k in range(self.n_states):
                for l in range(self.n_states):                    
                    if self.transition_mask[k, l]:
                        num = np.concatenate([ln_xi_s[p][1:, k, l] \
                            for p in range(n_sequences)]).astype(np_data_t)
                        den = np.concatenate([ln_gamma_s[p][:-1, k] \
                            for p in range(n_sequences)]).astype(np_data_t)
                        self.ln_transition_probs[k, l] = elogsum(num) - elogsum(den)
                    else: # If transition is impossible, due to model topology
                        self.ln_transition_probs[k, l] = LOG_ZERO

            self.ln_transition_probs = np.nan_to_num(self.ln_transition_probs)
            eexp2d(self.transition_probs, self.ln_transition_probs)
            np.asarray(self.transition_probs)[np.isnan(self.transition_probs)] = ZERO
            self.transition_probs /= np.sum(self.transition_probs, axis=1)[:, None]

            # Update emission parameters (for example, Gaussian parameters)
            X = np.concatenate(X_s, axis=0)
            gamma = np.concatenate(gamma_s, axis=0)
            self.update_emission_params(X, gamma)

    def fit(self, X_s, **kwargs):
        if isinstance(X_s, np.ndarray):
            X_s = [X_s]
        elif not isinstance(X_s, list):
            # TODO: raise exception
            pass
        # TODO: if empty list, raise exception
        first_seq = X_s[0]
        if len(first_seq.shape) == 1:
            self.n_features = 1
        else:
            self.n_features = first_seq.shape[1]
        # TODO: if parameters set by hand, do not pre-estimate parameters
        self.estimate_params(X_s)
        self.baum_welch(X_s, **kwargs)
    
    def log_likelihood(self, X):
        # TODO: CHECK X
        n_samples = len(X)
        lnf = self.emission_log_proba(X)
        cdef data_t[:, :] ln_alpha = np.zeros((n_samples, self.n_states))
        cdef data_t[:, :] ln_beta = np.zeros((n_samples, self.n_states))
        cdef data_t[:, :] ln_gamma = np.zeros((n_samples, self.n_states))
        cdef data_t[:, :, :] ln_xi = np.zeros(
            (n_samples, self.n_states, self.n_states))
        lnP = self.e_step(lnf, ln_alpha, ln_beta, ln_gamma, ln_xi, 0)

        gamma = np.empty_like(ln_gamma)
        eexp2d(gamma, ln_gamma)
        return lnP, gamma

    def decode(self, X):
        _, gamma = self.log_likelihood(X)
        return gamma.argmax(axis=1)

    def get_num_params(self):
        # TODO: what to do if self.fit has not yet been called ?
        n_emission_params = self.get_num_params_per_state() * self.n_states
        n_start_params = self.n_states - 1
        n_transition_params = self.n_states * (self.n_states - 1)
        return n_emission_params + n_start_params + n_transition_params

    def score(self, X, criterion='aic'):
        criterion = criterion.strip().lower()
        n = X.shape[0]

        # Compute "best" log-likelihood of sequence X
        # given the parameters of the model
        lnP, _ = self.log_likelihood(X)

        # Compute the model complexity
        k = self.get_num_params()

        # Compute information criterion
        if criterion == 'aic': # Akaike Information Criterion
            score_val = 2. * k - 2. * lnP
        elif criterion == 'aicc': # Akaike Information Criterion (corrected version)
            score_val = 2. * k - 2. * lnP + (2. * k * (k + 1.)) / (n - k - 1.)
        elif criterion == 'bic': # Bayesian Information Criterion
            score_val = k * elog(n) - lnP
        elif criterion == 'negloglh': # Negative log-likelihood
            score_val = -lnP
        else:
            raise NotImplementedError(
                "Unknown information criterion %s" % str(criterion))
        return score_val

    def sample(self, n_samples):
        # Initialize observation history and state history
        states = np.zeros(n_samples+1, dtype=np.int)
        observations = np.zeros((n_samples, self.n_features), dtype=np_data_t)

        # Randomly pick an initial state
        states[0] = np.random.choice(np.arange(self.n_states), p=self.initial_probs)

        # TODO: OPTIMIZATION WITH A NOGIL BLOCK
        for t in range(1, n_samples+1):
            state_id = states[t-1]
            observations[t-1, :] = self.sample_one_from_state(state_id)

            # Randomly pick next state w.r.t. transition probabilities
            states[t] = np.random.choice(
                np.arange(self.n_states), p=self.transition_probs[state_id])
        return states[:-1], observations
    
    def __str__(self):
        s = "HMM of type '%s'\n" % self.__class__.__name__
        s += "Topology '%s' with %i state(s)\n" % (self.arch, self.n_states) # TODO
        s += "Max number of free parameters: %i" % self.get_num_params()
        return s + "\n"
    
    def __repr__(self):
        return self.__str__()
    
    def __getstate__(self):
        return self.arch, self.n_features, \
            np.asarray(self.transition_mask), self.pi, self.a
    
    def __setstate__(self, state):
        self.arch, self.n_features, \
            self.transition_mask, self.pi, self.a = state
        self.n_states = self.pi.shape[0]
    
    property pi:
        def __get__(self):
            return np.asarray(self.initial_probs)
        def __set__(self, arr):
            self.initial_probs = np.asarray(arr, dtype=np_data_t)
            self.ln_initial_probs = np.empty_like(arr)
            elog1d(self.ln_initial_probs, self.initial_probs)

    property a:
        def __get__(self):
            return np.asarray(self.transition_probs)
        def __set__(self, arr):
            self.transition_probs = np.asarray(arr, dtype=np_data_t)
            self.ln_transition_probs = np.empty_like(arr)
            elog2d(self.ln_transition_probs, self.transition_probs)


cdef class GHMM(HMM):

    def __init__(self, n_states, arch='ergodic'):
        HMM.__init__(self, n_states, arch=arch)

    def estimate_params(self, X_s):
        first_seq = X_s[0]
        self.n_features = first_seq.shape[1]

        self.mu = np.empty(
            (self.n_states, self.n_features), dtype=np_data_t)
        self.sigma = np.empty(
            (self.n_states, self.n_features, self.n_features), dtype=np_data_t)

        if self.arch == 'linear' and False: # TODO
            # Make Change Point Detection
            """
            cpdetector  = GraphTheoreticDetector(
                n_keypoints=self.n_states-1, window_size=7)
            cpdetector.detectPoints(np.asarray(X, dtype=np.double))
            keypoint_indices = [0] + list(cpdetector.keypoints) + [len(X)]
            assert(len(keypoint_indices) == self.n_states+1)
            print(keypoint_indices)
            """
            cpd = BatchCPD(n_keypoints=self.n_states, window_padding=1,
                           cost_func=SUM_OF_SQUARES_COST, aprx_degree=2)
            cpd.detectPoints(X, X.mean(axis=0), np.cov(X.T))
            keypoint_indexes = cpd.getKeypoints()

            # Estimate start and transition probabilities
            self.transition_probs = np.zeros(
                (self.n_states, self.n_states), dtype=np.float)
            self.transition_probs[-1, -1] = 1.
            for i in range(self.n_states):
                a_ij = 1. / (keypoint_indices[i+1] - keypoint_indices[i])
                self.transition_probs[i, i+1] = a_ij
                self.transition_probs[i, i] = 1. - a_ij
            self.initial_probs[0] = 1.0

            # Estimate Gaussian parameters
            for i in range(self.n_states):
                segment = X[keypoint_indices[i]:keypoint_indices[i+1], :]
                self.mu[i] = segment.mean(axis=0)
                self.sigma[i] = np.cov(segment.T)
        elif self.arch == 'ergodic':
            # Apply clustering algorithm, and estimate Gaussian parameters
            X_concatenated = np.concatenate(X_s, axis=0)
            self.mu, indices = k_means(X_concatenated, self.n_states, n_runs=5)

            self.sigma = np.empty(
                (self.n_states, self.n_features, self.n_features), dtype=np_data_t)
            for i in range(self.n_states):
                tmp = np.cov(X_concatenated[indices == i].T)
                for j in range(self.n_features):
                    for k in range(self.n_features):
                        self.sigma[i, j, k] = tmp[j, k]

            # Estimate start and transition probabilities
            self.initial_probs = np.tile(1.0 / self.n_states, self.n_states)
            self.transition_probs = np.random.dirichlet([1.0] * self.n_states, self.n_states)
        else:
            pass # TODO: Exception: unknown arch

        self.ln_initial_probs = np.log(np.asarray(self.initial_probs))
        self.ln_transition_probs = np.log(np.asarray(self.transition_probs))

    def emission_log_proba(self, X):
        n_samples, n_features = X.shape[0], X.shape[1] 
        n_states = self.mu.shape[0]
        lnf = np.empty((n_samples, n_states), dtype=np_data_t)
        for k in range(n_states):
            lnf[:, k] = gaussian_log_proba(X, self.mu[k, :], self.sigma[k, :, :])
        return lnf
    
    def nan_to_zeros(self):
        np.asarray(self.mu)[np.isnan(self.mu)] = ZERO
        np.asarray(self.sigma)[np.isnan(self.sigma)] = ZERO

    def update_emission_params(self, X, gamma):
        self.nan_to_zeros()
        n_samples, n_features = X.shape[0], X.shape[1]
        for k in range(self.n_states):
            # Compute denominator for means and covariances
            posteriors = gamma[:, k]
            post_sum = posteriors.sum()
            norm = 1.0 / post_sum if post_sum != 0.0 else 1. # TODO

            # Update covariance matrix of state k
            # TODO: OPTIMIZATION WITH A NOGIL BLOCK
            covs = list()
            for t in range(n_samples):
                diff = X[t, :] - self.mu[k, :]
                covs.append(np.outer(diff, diff))
            covs = np.transpose(np.asarray(covs), (1, 2, 0))
            temp = np.sum(covs * posteriors, axis=2)

            # TODO: OPTIMIZATION WITH A NOGIL BLOCK
            for j in range(n_features):
                for l in range(n_features):
                    self.sigma[k, j, l] = temp[j, l]

            # Update mean of state k
            # TODO: OPTIMIZATION WITH A NOGIL BLOCK
            temp = np.dot(posteriors, X) * norm
            for j in range(n_features):
                self.mu[k, j] = temp[j]
        self.nan_to_zeros()
    
    cdef data_t[:] sample_one_from_state(self, int state_id) nogil:
        with gil: # TODO: GET RID OF PYTHON CALLS
            cholesky_sigma = np.linalg.cholesky(self.sigma[state_id, :, :])
            r = np.random.randn(self.n_features)
            return np.dot(r, cholesky_sigma.T) + self.mu[state_id, :]
    
    def get_num_params_per_state(self):
        # Number of parameters in mean vector
        n_mu = self.n_features
        # Number of parameters in half-vectorized covariance matrix
        n_sigma = self.n_features * (self.n_features + 1.) / 2.
        return n_mu + n_sigma
    
    def __getstate__(self):
        return HMM.__getstate__(self), np.asarray(self.mu), np.asarray(self.sigma)
    
    def __setstate__(self, state):
        base_state, self.mu, self.sigma = state
        HMM.__setstate__(self, base_state)

    property mu:
        def __get__(self):
            return np.asarray(self.mu)
        def __set__(self, arr):
            self.mu = np.asarray(arr)
    
    property sigma:
        def __get__(self):
            return np.asarray(self.sigma)
        def __set__(self, arr):
            self.sigma = np.asarray(arr)


cdef class GMMHMM(HMM):

    def __init__(self, n_states, arch='ergodic', n_components=3):
        HMM.__init__(self, n_states, arch=arch)
        self.n_components = n_components

    def estimate_params(self, X_s):
        self.n_features = X.shape[1]
        self.weights = np.empty((self.n_states, self.n_components), dtype=np_data_t)
        self.mu = np.empty(
            (self.n_states, self.n_components, self.n_features), dtype=np_data_t)
        self.sigma = np.empty(
            (self.n_states, self.n_components, self.n_features, self.n_features), dtype=np_data_t)
        
        # TODO: random initialization
        # TODO: make distinction between ergodic and linear

        X_concatenated = np.concatenate(X_s, axis=0)
        self.mu, indices = k_means(X_concatenated, self.n_states, n_runs=5)
        for i in range(self.n_states):
            cluster = X_concatenated[indices == i]
            sub_mu, sub_indices = k_means(cluster, self.n_components, n_runs=5)
            self.mu[i, :, :] = sub_mu
            for c in range(self.n_components):
                tmp = np.cov(cluster[sub_indices == c].T)
                for j in range(self.n_features):
                    for k in range(self.n_features):
                        self.sigma[i, c, j, k] = tmp[j, k]
                self.weights[i, c] = len(tmp)
            # Normalize weights per cluster
            self.weights[i, :] = self.weights[i, :] + np.sum(self.weights[i, :])

        # Estimate start and transition probabilities
        self.initial_probs = np.tile(1.0 / self.n_states, self.n_states)
        self.transition_probs = np.random.dirichlet([1.0] * self.n_states, self.n_states)
    
    def emission_log_proba(self, X):
        n_samples, n_features = X.shape[0], X.shape[1] 
        n_states = mu.shape[0]
        lnf = np.empty((n_samples, n_states), dtype=np_data_t)
        for k in range(n_states):
            lnf_by_component = np.empty(n_samples, self.n_components, dtype=np_data_t)
            for c in range(self.n_components):
                lnf_by_component[:, c] = gaussian_log_proba(
                    X, self.mu[k, c, :], self.sigma[k, c, :, :])
            
            # TODO: update lnf[:, k] with vectorized elogsum
        return lnf

    def nan_to_zeros(self):
        np.asarray(self.weights)[np.isnan(self.weights)] = ZERO
        np.asarray(self.mu)[np.isnan(self.mu)] = ZERO
        np.asarray(self.sigma)[np.isnan(self.sigma)] = ZERO
    
    def update_emission_params(self, X, gamma):
        n_sequences = len(X_s)
        _gamma = np.empty((n_samples, self.n_states, self.n_components), dtype=np_data_t)
        _gamma[:, :, :] = gamma
    
    cdef data_t[:] sample_one_from_state(self, int state_id) nogil:
        with gil: # TODO: GET RID OF PYTHON CALLS
            component = np.random.choice(
                np.arange(self.n_components), p=self.weights[state_id, :])
            cholesky_sigma = np.linalg.cholesky(
                self.sigma[state_id, component, :, :])
            r = np.random.randn(self.n_features)
            return np.dot(r, cholesky_sigma.T) + self.mu[state_id, component, :]

    def get_num_params_per_state(self):
        # Number of parameters in mean vector
        n_mu = self.n_features
        # Number of parameters in half-vectorized covariance matrix
        n_sigma = self.n_features * (self.n_features + 1.) / 2.
        return (n_mu + n_sigma) * self.n_components

    def __getstate__(self):
        return HMM.__getstate__(self), np.asarray(self.mu), \
            np.asarray(self.sigma), np.asarray(weights)
    
    def __setstate__(self, state):
        base_state, self.mu, self.sigma, self.weights = state
        self.n_components = self.weights.shape[1]
        HMM.__setstate__(self, base_state)

    property mu:
        def __get__(self):
            return np.asarray(self.mu)
        def __set__(self, arr):
            self.mu = np.asarray(arr)
    
    property sigma:
        def __get__(self):
            return np.asarray(self.sigma)
        def __set__(self, arr):
            self.sigma = np.asarray(arr)
    
    property weights:
        def __get__(self):
            return np.asarray(self.weights)
        def __set__(self, arr):
            self.weights = np.asarray(arr)


cdef class MHMM(HMM):
    """ Multinomial Hidden Markov Model

    Attributes:
        n_unique (int):
            Number of unique values that can be observed
        proba (:obj:`np.ndarray`):
            Array of shape (n_states, n_unique) where proba[i, j]
            is the probability of observing value j when being
            in hidden state i.
    """

    def __init__(self, n_states, arch='ergodic'):
        HMM.__init__(self, n_states, arch=arch)

    def estimate_params(self, X_s):
        """ Pre-estimates model parameters

        Args:
            X_s (list):
                List of sequences, where sequence i is a NumPy array
                of shape (n_samples_i, n_features)
        """
        # TODO: CHECK X
        X_concatenated = np.concatenate(X_s, axis=0)
        # self.n_unique = len(np.unique(np.squeeze(X_concatenated)))
        self.n_unique = np.max(X_concatenated) + 1
        self.proba = np.random.rand(self.n_states, self.n_unique).astype(np_data_t)
        self.proba /= np.sum(self.proba, axis=1)[:, None]
        # TODO: estimation algorithms

        self.initial_probs = np.random.rand(self.n_states).astype(np_data_t)
        self.initial_probs /= np.sum(self.initial_probs)
        self.ln_initial_probs = np.log(self.initial_probs)

        self.transition_probs = np.random.rand(self.n_states, self.n_states).astype(np_data_t)
        self.transition_probs /= np.sum(self.transition_probs, axis=1)[:, None]
        self.ln_transition_probs = np.log(self.transition_probs)

    def emission_log_proba(self, X):
        """ Computes emission log-probabilities of a given sequence according to the
        parameters of the multinomial distributions of each state.

        Args:
            X (:obj:`np.ndarray`):
                Sequence of observed samples
        
        Returns:
            :obj:`np.ndarray`:
                Emission log-probabilities of given sequence
        """
        cdef int n_samples = X.shape[0]
        cdef data_t[:, :] lnf = np.empty((n_samples, self.n_states), dtype=np_data_t)
        cdef cnp.int_t[:] _X = X.astype(np.int)
        with nogil:
            for k in range(self.n_states):
                for t in range(n_samples):
                    lnf[t, k] = libc.math.log(self.proba[k, <int>_X[t]])
        return np.asarray(lnf)

    def nan_to_zeros(self):
        np.asarray(self.proba)[np.isnan(self.proba)] = ZERO
    
    def update_emission_params(self, X, gamma):
        self.nan_to_zeros()

        # TODO: OPTIMIZATION WITH A NOGIL BLOCK
        for k in range(self.n_states):
            posteriors = gamma[:, k]
            post_sum = posteriors.sum()
            norm = 1.0 / post_sum if post_sum != 0.0 else 1.
            for i in range(self.n_unique):
                self.proba[k, i] = np.dot(posteriors, X == i) * norm

        self.nan_to_zeros()
    
    cdef data_t[:] sample_one_from_state(self, int state_id) nogil:
        """ Sample a multinomial distribution, where the distribution
        is associated to current hidden state.

        Args:
            state_id (int):
                Hidden state associated to the distribution to sample

        Returns:
            data_t:
                Random sample
        """

        with gil: # TODO: REMOVE GIL BLOCK
            weights = self.proba[state_id]
            return np.random.choice(np.arange(self.n_unique), p=weights)

    def get_num_params_per_state(self):
        """ Returns an upper bound on the number of free parameters
        for a single hidden state.

        The number of required parameters is equal to n_unique - 1 since
        any emission probability vector sums to one.

        Returns:
            int:
                Upper bound on the number of free parameters per state
        """
        # Number of free parameters in proba vector
        n_params = self.n_unique - 1
        return n_params

    def __getstate__(self):
        return HMM.__getstate__(self), np.asarray(self.proba)
    
    def __setstate__(self, state):
        base_state, self.proba = state
        self.n_unique = self.proba.shape[1]
        HMM.__setstate__(self, base_state)

    property proba:
        def __get__(self):
            return np.asarray(self.proba)
        def __set__(self, arr):
            self.proba = np.asarray(arr)
