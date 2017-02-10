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
        bint is_incremental
        double min_threshold
        size_t max_height
        size_t n_classes
        size_t max_nodes
        int partitioning
        data_t nan_value

    struct Tree:
        Node* root
        size_t n_nodes
        size_t n_classes
        size_t n_features
        TreeConfig* config

    struct Density:
        bint    is_categorical
        data_t  split_value
        data_t* quartiles
        data_t* deciles
        data_t* percentiles
        size_t* counters_left
        size_t* counters_right
        size_t* counters_nan

    struct Splitter:
        Node* node
        size_t n_instances
        size_t feature_id
        size_t n_features
        target_t* targets
        data_t nan_value

    Node* newNode(size_t n_classes)
    inline float ShannonEntropy(float probability)
    inline float GiniCoefficient(float probability)
    Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                              size_t n_classes, data_t nan_value)
    Tree* ID3(data_t* data, target_t* targets, size_t n_instances, size_t n_features,
              TreeConfig* config)
    float* classify(data_t* data, size_t n_instances, size_t n_features,
                    Tree* tree, TreeConfig* config)

cdef extern from "id4_.h":
    void ID4_Update(Tree* tree, data_t* data, target_t* targets, 
            size_t n_instances, size_t n_features, size_t n_classes, data_t nan_value)