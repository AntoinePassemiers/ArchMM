#ifndef ID3__H_
#define ID3__H_

#include "queue_.h"

#define NO_FEATURE -1
#define NO_INSTANCE 0
#define NO_SPLIT_VALUE INFINITY

typedef double data_t;

struct Node {
    int feature_id;
    size_t* counters;
    size_t n_instances;
    double score;
    double split_value;
    struct Node* left_child;
    struct Node* right_child;
};

struct TreeConfig {
    size_t max_height;
    size_t n_classes;
    size_t max_nodes;
};

struct Tree {
    struct Node root;
    size_t n_nodes;
    size_t n_classes;
    size_t n_features;
    struct TreeConfig* config;
};

struct Density {
    data_t split_value;
    data_t* quartiles;
    data_t* deciles;
    data_t* percentiles;
    data_t* high_precision_partition;
};

struct Splitter {
    struct Node* node;
    size_t n_instances;
    size_t feature_id;
    size_t n_features;
};

struct Node* newNode(size_t n_classes);

inline float ShannonEntropy(float* probabilities, size_t n_classes);

inline float GiniCoefficient(float* probabilities, size_t n_classes);

struct Density* computeDensities(data_t** data, size_t n_instances, 
                                 size_t n_features, int use_high_precision,
                                 data_t nan_value);

#endif // ID3__H_