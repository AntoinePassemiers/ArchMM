#ifndef ID3__H_
#define ID3__H_

#include "queue_.h"
#include <stddef.h>

#define NO_FEATURE       -1
#define NO_INSTANCE       0
#define NO_SPLIT_VALUE    INFINITY
#define NUM_SPLIT_LABELS  3
#define COST_OF_EMPTINESS INFINITY

#define QUARTILE_PARTITIONING       1
#define DECILE_PARTITIONING         2
#define PERCENTILE_PARTITIONING     3

struct Node {
    int id;
    int feature_id;
    size_t* counters;
    size_t n_instances;
    double score;
    double split_value;
    struct Node* left_child;
    struct Node* right_child;
};

struct TreeConfig {
    bint is_incremental;
    double min_threshold;
    size_t max_height;
    size_t n_classes;
    size_t max_nodes;
    int partitioning;
    data_t nan_value;
};

struct Tree {
    struct Node* root;
    size_t n_nodes;
    size_t n_classes;
    size_t n_features;
    struct TreeConfig* config;
};

struct Density {
    bint    is_categorical;
    data_t  split_value;
    data_t* quartiles;
    data_t* deciles;
    data_t* percentiles;
    size_t* counters_left;
    size_t* counters_right;
    size_t* counters_nan;
};

struct Splitter {
    struct Node* node;
    size_t n_instances;
    data_t* partition_values;
    size_t n_classes;
    size_t* belongs_to;
    size_t feature_id;
    size_t n_features;
    target_t* targets; 
    data_t nan_value;
};

struct Node* newNode(size_t n_classes);

extern inline float ShannonEntropy(float probability);

extern inline float GiniCoefficient(float probability);

struct Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                                 size_t n_classes, data_t nan_value);

double evaluatePartitions(data_t* data, struct Density* density,
                          struct Splitter* splitter, size_t k);

extern inline double getFeatureCost(struct Density* density, size_t n_classes);

double evaluateByThreshold(struct Splitter* splitter, struct Density* density, 
                           data_t* data, int partition_value_type);

struct Tree* ID3(data_t* data, target_t* targets, size_t n_instances, size_t n_features,
                 struct TreeConfig* config);

float* classify(data_t* data, size_t n_instances, size_t n_features,
                struct Tree* tree, struct TreeConfig* config);

#endif // ID3__H_