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


cpdef int QUARTILE_PARTITIONING   = 1
cpdef int DECILE_PARTITIONING     = 2
cpdef int PERCENTILE_PARTITIONING = 3

cdef class ClassificationTree:
    cdef Tree* tree
    cdef TreeConfig* config
    def __init__(self, min_threshold = 1.0e-6, max_height = 50, max_nodes = 4500,
                 partitioning = 100):
        self.tree = NULL
        self.config = <TreeConfig*>malloc(sizeof(TreeConfig))
        self.config.is_incremental = False
        self.config.min_threshold = min_threshold
        self.config.max_nodes = max_nodes
        self.config.partitioning = partitioning
        self.config.nan_value = -1
        self.config.max_height = max_height

    property is_incremental:
        def __get__(self): return self.config.is_incremental
    property min_threshold:
        def __get__(self): return self.config.min_threshold
    property max_nodes:
        def __get__(self): return self.config.max_nodes
    property nan_value:
        def __get__(self): return self.config.nan_value
    property max_height:
        def __get__(self): return self.config.max_height
    property n_nodes:
        def __get__(self):
            if self.tree == NULL:
                return 0
            return self.tree.n_nodes
    property partitioning:
        def __get__(self): 
            if self.config.partitioning == QUARTILE_PARTITIONING:
                return "quartiles"
            elif self.config.partitioning == DECILE_PARTITIONING:
                return "deciles"
            else:
                return "percentiles"


    def applyID3(self, data, targets):
        data = format_data(data)
        ensure_array(targets, ndim = 1, msg = "Targets must be a 1d array")
        assert(np.min(targets) == 0) 
        self.config.n_classes = np.max(targets) + 1
        cdef size_t n_instances = data.shape[1]
        cdef size_t n_features = data.shape[2]
        assert(n_instances == len(targets))
        cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
        cdef int[:] btargets = np.asarray(targets, dtype = np.int)
        self.tree = ID3(<data_t*>&bdata[0, 0], <target_t*>&btargets[0],
                        n_instances, n_features, self.config)

    def update(self, data, targets):
        data = format_data(data, ndim = 3)
        ensure_array(targets, ndim = 1, msg = "Targets must be a 1d array")
        assert((0 <= targets).all() and (targets < self.config.n_classes).all())
        cdef size_t n_instances = data.shape[1]
        cdef size_t n_features = data.shape[2]
        assert(n_instances == len(targets))
        cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
        cdef int[:] btargets = np.asarray(targets, dtype = np.int)
        ID4_Update(self.tree, <data_t*>&bdata[0, 0], <target_t*>&btargets[0], 
                   n_instances, n_features, self.config.n_classes, self.config.nan_value)
        
    def classify(self, data):
        data = format_data(data, ndim = 3)
        cdef size_t n_instances = data.shape[1]
        cdef size_t n_features = data.shape[2]
        cdef size_t n_classes = self.config.n_classes
        cdef double[::view.strided, ::1] bdata = data.data[0, :, :]
        cdef float* predictions = classify(<data_t*>&bdata[0, 0], n_instances, 
                        n_features, self.tree, self.config)
        return np.asarray(<float[:n_instances, :n_classes]>predictions)

    def deleteTree(self): # TODO : Switch to recursive function
        cdef Queue* queue = newQueue()
        cdef Node* current_node = self.tree.root
        free(self.tree)
        deleteSubtree(current_node)
        self.tree = NULL

    def __dealloc__(self):
        self.deleteTree()
        free(self.config)
            
    def save(self, filepath):
        cdef size_t i
        cdef size_t num_nodes = 0
        cdef FILE* ptr_fw = fopen(<char*>filepath, "w")
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