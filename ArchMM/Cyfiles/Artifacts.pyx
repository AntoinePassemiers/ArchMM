# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport * 

""" TODO
- Detect drop-outs sequences
https://www.quora.com/How-can-I-estimate-the-parameters-of-a-discrete-time-HMM-when-some-observations-are-missing
- Detect outliers
"""

DEF DEFAULT_MISSING_VALUE = -87.89126015 

cpdef cnp.ndarray getMissingValuesIndexes(cnp.ndarray data, double missing_value):
    return np.where(data == missing_value)[0]

"""
>>> c=theano.tensor.as_tensor_variable(c)
>>> inf_mask = T.isinf(c)
>>> nan_mask = T.isnan(c)
>>> inf_idx = inf_mask.nonzero()
>>> nan_idx = nan_mask.nonzero()
>>>
>>> c_without_inf = theano.tensor.set_subtensor(c[inf_idx], -999)
>>> c_without_inf_nan = theano.tensor.set_subtensor(c_without_inf[nan_idx],
"""