# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp
cnp.import_array()


ctypedef cnp.float32_t prob_t
ctypedef cnp.float32_t ln_prob_t

cdef struct subHHMM:
    size_t current_state
    prob_t[:, :] A
    void** concrete_children