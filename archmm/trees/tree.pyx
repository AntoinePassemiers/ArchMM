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
        self.config.use_high_precision = use_high_precision
        self.config.nan_value = -1
        self.config.max_height = max_height

    def applyID3(self, data, targets):
        data = format_data(data)
        ensure_array(targets, ndim = 1, msg = "Targets must be a 1d array")
        assert(np.min(targets) == 0) 
        self.config.n_classes = np.max(targets) + 1
        cdef size_t n_instances = data.shape[1]
        cdef size_t n_features = data.shape[2]
        assert(n_instances == len(targets))
        cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
        cdef int[:] btargets = targets
        self.tree = ID3(<data_t*>&bdata[0, 0], <target_t*>&btargets[0], 
                        n_instances, n_features, self.config) 
    def update(self, data, targets):
        # TODO
        data = format_data(data)
        ensure_array(targets, ndim = 1, msg = "Targets must be a 1d array")
        assert((0 <= targets).all() and (targets < self.config.n_classes).all())
        cdef size_t n_instances = data.shape[1]
        cdef size_t n_features = data.shape[2]
        assert(n_instances == len(targets))
        cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
        cdef int[:] btargets = targets
        ID4_Update(self.tree, <data_t*>&bdata[0, 0], <target_t*>&btargets[0], 
                   n_instances, n_features, self.config.n_classes, self.config.nan_value)
    def classify(self, data):
        data = format_data(data)
        cdef size_t n_instances = data.shape[1]
        cdef size_t n_features = data.shape[2]
        cdef size_t n_classes = self.config.n_classes
        cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
        cdef float* predictions = classify(<data_t*>&bdata[0, 0], n_instances, 
                        n_features, self.tree, self.config)
        return np.asarray(<float[:n_instances, :n_classes]>predictions)

    def deleteTree(self):
        cdef Queue* queue = newQueue()
        cdef Node* current_node = self.tree.root
        free(self.tree)
        enqueue(queue, current_node)
        while not isQueueEmpty(queue):
            current_node = <Node*>dequeue(queue)
            if current_node.left_child != NULL:
                enqueue(queue, <void*>current_node.left_child)
            if current_node.right_child != NULL:
                enqueue(queue, <void*>current_node.right_child)
            free(current_node.counters)
            free(current_node)
        free(queue)
        self.tree = NULL
    def __dealloc__(self):
        free(self.config)
    property n_nodes:
        def __get__(self):
            if self.tree == NULL:
                return 0
            return self.tree.n_nodes
            
    def save(self):
        cdef size_t i
        cdef size_t num_nodes = 0
        cdef FILE* ptr_fw = fopen("C://Users/Xanto183/git/ArchMM/archmm/trees/test.arctree", "w")
        if ptr_fw == NULL:
            raise IOError("Could not open the file")
        cdef Queue* queue = newQueue()
        cdef Node* current_node = self.tree.root
        printf("\nTotal number of nodes : %i\n", self.tree.n_nodes);
        enqueue(queue, current_node)
        fprintf(ptr_fw, "%i\n", self.tree.n_nodes)
        while not isQueueEmpty(queue):
            current_node = <Node*>dequeue(queue)
            printf("\nId : %i\n", current_node.id)
            printf("Address : %p\n", current_node)
            printf("Left child address : %p\n", current_node.left_child)
            printf("Right child address : %p\n", current_node.right_child)
            printf("Number of instances : %i\n", current_node.n_instances)
            printf("Score : %f\n", current_node.score)
            printf("Split point : %f\n", current_node.split_value)
            printf("Feature : %i\n", current_node.feature_id)
            printf("Counters : %i, %i\n", current_node.counters[0], current_node.counters[1])
            fprintf(ptr_fw, "%i,%f,%i", 
                    current_node.id, current_node.split_value, current_node.feature_id)
            num_nodes += 1
            if current_node.left_child != NULL:
                enqueue(queue, <void*>current_node.left_child)
                fprintf(ptr_fw, ",%i", current_node.left_child.id)
            if current_node.right_child != NULL:
                enqueue(queue, <void*>current_node.right_child)
                fprintf(ptr_fw, ",%i", current_node.right_child.id)
            if num_nodes < self.tree.n_nodes:
                fprintf(ptr_fw, "\n")
        free(queue)
        fclose(ptr_fw)

cpdef getDensities(data_array):
    data = format_data(data_array)
    cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
    cdef size_t n_instances = data.shape[1]
    cdef size_t n_features = data.shape[2]
    cdef size_t i, j
    cdef data_t* pdata = <data_t*>&bdata[0, 0]
    cdef Density* density = computeDensities(pdata, n_instances, n_features, 3, 0, -1)
    densities = list()
    cdef cnp.double_t[::1] quartiles, deciles, percentiles
    for i in range(n_features):
        quartiles = np.ascontiguousarray(np.empty(4, dtype = np.double))
        deciles = np.ascontiguousarray(np.empty(10, dtype = np.double))
        percentiles = np.ascontiguousarray(np.empty(100, dtype = np.double))
        memcpy(&quartiles[0], density[i].quartiles, 4 * sizeof(cnp.double_t))
        memcpy(&deciles[0], density[i].deciles, 10 * sizeof(cnp.double_t))
        memcpy(&percentiles[0], density[i].percentiles, 100 * sizeof(cnp.double_t))
        densities.append({
            "quartiles" : np.asarray(quartiles),
            "deciles" : np.asarray(deciles),
            "percentiles" : np.asarray(percentiles)
        })
    return densities