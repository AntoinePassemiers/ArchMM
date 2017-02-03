# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *


cpdef cnp.ndarray getMissingValuesIndexes(cnp.ndarray data, double missing_value)