#ifndef ID3__H_
#define ID3__H_

#include "queue_.h"

#define NO_FEATURE       -1
#define NO_INSTANCE       0
#define NO_SPLIT_VALUE    INFINITY
#define NUM_SPLIT_LABELS  3
#define COST_OF_EMPTINESS INFINITY

#define QUARTILE_PARTITIONING       1
#define DECILE_PARTITIONING         2
#define PERCENTILE_PARTITIONING     3
#define HIGH_PRECISION_PARTITIONING 4

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
    double min_threshold;
    size_t max_height;
    size_t n_classes;
    size_t max_nodes;
    bint use_high_precision;
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
    data_t  split_value;
    data_t* quartiles;
    data_t* deciles;
    data_t* percentiles;
    data_t* high_precision_partition;
    size_t* counters_left;
    size_t* counters_right;
    size_t* counters_nan;
};

struct Splitter {
    struct Node* node;
    size_t n_instances;
    size_t feature_id;
    size_t n_features;
    target_t* targets; 
    data_t nan_value;
};

struct Node* newNode(size_t n_classes);

extern inline float ShannonEntropy(float probability);

inline float GiniCoefficient(float probability);

struct Density* computeDensities(data_t* data, size_t n_instances, size_t n_features,
                                 size_t n_classes, int use_high_precision, data_t nan_value);


double evaluateByThreshold(struct Splitter* splitter, struct Density* density, 
                                  data_t* data, size_t* belongs_to, size_t n_classes,
                                  int partition_value_type, size_t n_partition_values);

extern inline double getFeatureCost(struct Density* density, size_t n_classes);

double evaluatePartitions(data_t* data, struct Density* density,
                                 data_t* partition_values, struct Splitter* splitter, 
                                 size_t n_classes, size_t* belongs_to, size_t k);

struct Tree* ID3(data_t* data, target_t* targets, size_t n_instances, size_t n_features,
                 struct TreeConfig* config);

float* classify(data_t* data, size_t n_instances, size_t n_features,
                struct Tree* tree, struct TreeConfig* config);

#endif // ID3__H_