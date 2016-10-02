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