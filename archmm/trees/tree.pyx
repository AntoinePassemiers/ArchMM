# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport *

DEF BAD_PREDICTION = -1.0
DEF NO_SPLIT_POINT = -1.0
DEF MISSING_VALUE  = -1.0

ctypedef cnp.float32_t data_t

cdef struct Node:
    int         id
    char*       value
    int         feature
    float       pplus # TODO : must be a vector of probabilities (multi-class classification)
    data_t      split_point
    Node*       left_child
    Node*       right_child

cdef struct Tree:
    Node*  root
    size_t num_nodes

cdef Node* createNode():
    cdef Node* node = <Node*>malloc(sizeof(Node))
    node.feature = -1
    node.pplus = BAD_PREDICTION
    node.value = "null"
    node.split_point = NO_SPLIT_POINT
    node.id = 0
    node.left_child = NULL
    node.right_child = NULL
    return node