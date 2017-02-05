#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NO_FEATURE -1
#define NO_INSTANCE 0


struct Node {
    int feature_id;
    size_t* counters;
    size_t n_instances;
    double score;
    double split_value;
    struct Node* left_child;
    struct Node* right_child;
};

struct Tree {
    struct Node root;
    size_t n_classes;
    size_t n_features;
};

struct Node* newNode(size_t n_classes) {
    struct Node* node = (struct Node*) malloc(sizeof(struct Node));
    node->feature_id = NO_FEATURE;
    node->counters = (size_t*) malloc(n_classes * sizeof(size_t));
    node->n_instances = NO_INSTANCE;
    node->score = INFINITY;
}

inline float ShannonEntropy(float* probabilities, size_t n_classes) {
    float entropy = 0.0;
    for (int i = 0; i < n_classes; i++) {
        entropy -= probabilities[i] * log2(probabilities[i]);
    }
    return entropy;
}

inline float GiniCoefficient(float* probabilities, size_t n_classes) {
    float gini = 0.0;
    for (int i = 0; i < n_classes; i++) {
        gini += probabilities[i] * probabilities[i];
    }
    gini = 1.0 - gini;
    return gini;
}