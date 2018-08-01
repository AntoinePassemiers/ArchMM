# -*- coding: utf-8 -*-
# exceptions.py
# author: Antoine Passemiers


class UnknownTopology(Exception):
    """ Exception raised when a topology is not of type
    :class:`archmm.topology.Topology` or a name of a known
    topology.
    """
    pass


class UnknownCriterion(Exception):
    """ Exception raised when a criterion name is unknown. """
    pass


class UntrainedModelError(Exception):
    """ Exception raised when a certain method is called before
    the model has been trained. Such method is requires 
    information available only after training the model.
    """
    pass
