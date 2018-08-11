# -*- coding: utf-8 -*-
# utils.py
# author: Antoine Passemiers

import numpy as np


def abstractpythonmethod(func):
    def func_wrapper(*args):
        raise NotImplementedError(
            "%s abstract method must be implemented" % func.__name__
        )
    func_wrapper.__name__ = func.__name__
    return func_wrapper


def binarize_labels(y, n_classes):
    """ Transforms a N-valued vector of labels into a binary vector.
    Args:
        y (:obj:`np.ndarray`):
            Vector of classes.
        n_classes (int):
            Number of classes.
    Returns:
        :obj:`np.ndarray`:
            binary_y:
                Binary matrix of shape (n_instances, n_classes)
                s.t. `binary_y[i,j] == 1` iff `y[i] == j`.
    Example:
        >>> y = np.array([0, 1, 0, 0, 1, 1, 0])
        >>> binarize_labels(y)
        array([[1, 0],
               [0, 1],
               [1, 0],
               [1, 0],
               [0, 1],
               [0, 1],
               [1, 0]])
    """
    binary_y = np.empty((len(y), n_classes), dtype=np.int)
    for c in range(n_classes):
        binary_y[:, c] = (y == c)
    return binary_y
