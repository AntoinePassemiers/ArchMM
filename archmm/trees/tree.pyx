# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdlib cimport *
from libc.stdio cimport *

cdef extern from "id3_.h":
    struct Node:
        pass
    struct Tree:
        pass
    struct Density:
        pass
    Node* newNode(size_t n_classes)
    inline float ShannonEntropy(float* probabilities, size_t n_classes)
    inline float GiniCoefficient(float* probabilities, size_t n_classes)

cdef Node* node = newNode(4)