# -*- coding: utf-8 -*-
# distutils: language=c

ctypedef int generic

cdef extern from "utils_.h":
    ctypedef generic bint

cdef extern from "id3_.h":
    ctypedef generic data_t
    ctypedef generic target_t

    struct Node:
        int id
        int feature_id
        size_t* counters
        size_t n_instances
        double score
        double split_value
        Node* left_child
        Node* right_child
    struct TreeConfig:
        double min_threshold
        size_t max_height
        size_t n_classes
        size_t max_nodes
        bint use_high_precision
        data_t nan_value
    struct Tree:
        Node root
        size_t n_nodes
        size_t n_classes
        size_t n_features
        TreeConfig* config
    struct Density:
        data_t split_value
        data_t* quartiles
        data_t* deciles
        data_t* percentiles
        data_t* high_precision_partition
        size_t* counters_left
        size_t* counters_right
        size_t* counters_nan
    struct Splitter:
        Node* node
        size_t n_instances
        size_t total_left
        size_t total_right
        size_t feature_id
        size_t n_features
        target_t* targets
        data_t nan_value

    Node* newNode(size_t n_classes)
    inline float ShannonEntropy(float probability)
    inline float GiniCoefficient(float probability)
    Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                              size_t n_classes, int use_high_precision, data_t nan_value)

    inline double evaluateByThreshold(Splitter* splitter, Density* density, 
                                      data_t* data, size_t* belongs_to, size_t n_classes,
                                      int partition_value_type, size_t n_partition_values)
    Tree* ID3(data_t* data, target_t* targets, size_t n_instances, size_t n_features,
              TreeConfig* config)