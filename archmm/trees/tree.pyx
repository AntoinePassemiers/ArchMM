# -*- coding: utf-8 -*-
# distutils: language=c

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport memcpy
from cython cimport view

from archmm.format_data import *
from archmm.queue cimport *

cdef class ClassificationTree:
    cdef Tree* tree
    cdef TreeConfig* config
    def __init__(self, min_threshold = 1.0e-6, max_height = 50, max_nodes = 4500,
                 use_high_precision = False):
        self.tree = NULL
        self.config = <TreeConfig*>malloc(sizeof(TreeConfig))
        self.config.min_threshold = min_threshold
        self.config.max_nodes = max_nodes
        self.config.n_classes = 3 # TODO : infer n_classes from targets
        self.config.use_high_precision = use_high_precision
        self.config.nan_value = -1
        self.config.max_height = max_height
    def applyID3(self, data, targets):
        data = format_data(data)
        ensure_array(targets, ndim = 1, msg = "Targets must be a 1d array")
        cdef size_t n_instances = data.shape[1]
        cdef size_t n_features = data.shape[2]
        assert(n_instances == len(targets))
        cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
        cdef int[:] btargets = targets
        self.tree = ID3(<data_t*>&bdata[0, 0], <target_t*>&btargets[0], 
                        n_instances, n_features, self.config) 
    def __dealloc__(self):
        pass
    property n_nodes:
        def __get__(self):
            if self.tree == NULL:
                return 0
            return self.tree.n_nodes
    def save(self):
        cdef size_t i
        cdef size_t num_nodes = 0
        cdef FILE* ptr_fw = fopen("C://Users/Xanto183\/git/ArchMM/archmm/trees/test.arctree", "w")
        if ptr_fw == NULL:
            raise IOError("Could not open the file")
        cdef Queue* queue = newQueue()
        cdef Node* current_node = &self.tree.root
        enqueue(queue, current_node)
        fprintf(ptr_fw, "%i\n", self.tree.n_nodes)
        while not isQueueEmpty(queue):
            current_node = <Node*>dequeue(queue)
            
            printf("\nId : %i\n", current_node.id)
            printf("Address : %p\n", &current_node)
            printf("Split point : %f\n", current_node.split_value)
            printf("Feature : %i\n", current_node.feature_id)
            printf("a\n");
            fprintf(ptr_fw, "%i,%f,%i", 
                    current_node.id, current_node.split_value, current_node.feature_id)
            num_nodes += 1
            printf("b\n");
            if current_node.left_child != NULL:
                enqueue(queue, &current_node.left_child)
                fprintf(ptr_fw, ",%i", current_node.left_child.id)
            if current_node.right_child != NULL:
                enqueue(queue, &current_node.right_child)
                fprintf(ptr_fw, ",%i", current_node.right_child.id)
            printf("c\n");
            if num_nodes < self.tree.n_nodes:
                fprintf(ptr_fw, "\n")
            printf("d\n");
        fclose(ptr_fw)

cpdef getDensities(data_array):
    data = format_data(data_array)
    cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
    cdef size_t n_instances = data.shape[1]
    cdef size_t n_features = data.shape[2]
    cdef size_t i, j
    cdef data_t* pdata = <data_t*>&bdata[0, 0]
    cdef Density* density = computeDensities(pdata, n_instances, n_features, 3, 1, 0)
    cdef cnp.double_t[::1] quartiles = np.ascontiguousarray(np.empty(4, dtype = np.double))
    cdef cnp.double_t[::1] deciles = np.ascontiguousarray(np.empty(10, dtype = np.double))
    cdef cnp.double_t[::1] percentiles = np.ascontiguousarray(np.empty(100, dtype = np.double))

    memcpy(&quartiles[0], density[1].quartiles, 4 * sizeof(cnp.double_t))
    memcpy(&deciles[0], density[1].deciles, 10 * sizeof(cnp.double_t))
    memcpy(&percentiles[0], density[1].percentiles, 100 * sizeof(cnp.double_t))
    densities = {
        "quartiles" : np.asarray(quartiles),
        "deciles" : np.asarray(deciles),
        "percentiles" : np.asarray(percentiles)
    }
    return densities