# -*- coding: utf-8 -*-
# utils.py
# author: Antoine Passemiers


def abstractpythonmethod(func):
    def func_wrapper(*args):
        raise NotImplementedError(
            "%s abstract method must be implemented" % func.__name__
        )
    func_wrapper.__name__ = func.__name__
    return func_wrapper
