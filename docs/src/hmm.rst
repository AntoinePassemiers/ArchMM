Hidden Markov Models
--------------------

Because ArchMM offers multi-sequence support, an additional level of formalism has been
added to take the sequence indexing into account.

Let :math:`a_{ij}` be the probability that the model switches from state :math:`i` to
state :math:`j` at any time. This allows us to define a matrix :math:`A \in \mathbb{R}^{n \times n}`
with rows summing to one and :math:`n` being the number of hidden states. Depending on the model
topology, some elements of :math:`A` may be equal to zero. For example, a left-to-right topology
constrains all elements that don't have a multi-index of the form :math:`(i, i)` or :math:`(i, i+1)`
to be equal to zero: the model can either stay in the same state or move to the next one.

.. math::
    a_{ij} = P(X_{t, p} = j | X_{t-1, p} = i)

The model can make transitions between states but still must be able to start in one of them.
Let :math:`\pi_i` be the probability that the model starts in state :math:`i` when
generating a new sequence.

.. math::
    \pi_i = P(X_{1, p} = i)

More importantly, a Hidden Markov Model should be capable of emitting an observation using the distribution
associated to current hidden state. Let's define :math:`b_j(y_i)` as the probability of emitting
the observed sample :math:`y_i` when being in state :math:`j`. Each hidden state has a dedicated
distribution with different parameter values.

.. math::
    b_j(y_i) = P(Y_{t, p} = y_i | X_{t, p} = j)

Before to jump to the learning part, let's introduce the 3 assumptions made by HMMs in order to make
the optimization algorithms computationally efficient:

- The implementation relies on the first-order Markov assumption, which implies that the probability
  of being in any state at current time only depends on what the previous state was.
- The transition probabilites between states are stationarity, meaning that they are time-independent.
- The observed samples are independent, meaning that they are not correlated over time.

The first assumption is the strongest one since it drastically simplifies the way the log-likelihood
of the data is computed. The log-likelihood is the logarithm of the probability of obtaining the given
sequences using the current parameters of the model. Because the probability of a long sequence is
not numerically stable (such a function approaches zero very quickly), its logarithm is used instead.
Plus, it has interesting statistical and computational properties.
It can be formalized as follows:

.. math::
    log(P(X | \theta)) = log \big[ TODO


Forward procedure
#################

.. math::
    \alpha_i(t, p) = P(Y_{1, p} = y_1, \ldots, Y_{t, p} = y_{t, p}, X_{t, p} = i | \theta)

.. math::
    \alpha_i(1, p)     &= \pi_i b_i(y_{1, p}) \\
    \alpha_i(t + 1, p) &= b_i(y_{t+1, p}) \sum\limits_{j=1}^{N} \alpha_j(t, p) a_{ji}

Backward procedure
##################

.. math::
    \beta_i(t, p) = P(Y_{t+1, p} = y_{t+1, p}, \ldots, Y_{T_p, p} = y_{T_p, p} | X_{t, p} = i, \theta)

.. math::
    \beta_i(T_p, p) &= 1 \\
    \beta_i(t, p)   &= \sum\limits_{j=1}^N \beta_j(t+1, p) a_{ij} b_j(y_{t+1, p})