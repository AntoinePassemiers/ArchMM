#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "id3_.h"

struct Node* newNode(size_t n_classes) {
    struct Node* node = (struct Node*) malloc(sizeof(struct Node));
    node->feature_id = NO_FEATURE;
    node->counters = (size_t*) malloc(n_classes * sizeof(size_t));
    node->n_instances = NO_INSTANCE;
    node->score = INFINITY;
    node->split_value = NO_SPLIT_VALUE;
    node->left_child = NULL;
    node->right_child = NULL;
    return node;
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

struct Density* computeDensities(data_t** data, size_t n_instances, 
                                 size_t n_features, int use_high_precision,
                                 data_t nan_value) {
    size_t s = sizeof(data_t);
    struct Density* densities = (struct Density*) malloc(n_features * sizeof(struct Density));
    data_t* sorted_values = (data_t*) malloc(n_instances * s);
    for (int f = 0; f < n_features; f++) {
        densities[f].quartiles = (data_t*) malloc(4 * s);
        densities[f].deciles = (data_t*) malloc(10 * s);
        densities[f].percentiles = (data_t*) malloc(100 * s);
        if (use_high_precision) {
            densities[f].high_precision_partition = (data_t*) malloc(use_high_precision * s);
        }
        // Putting nan values aside
        size_t n_acceptable = 0;
        for (int i = 0; i < n_instances; i++) {
            if (data[i][f] != nan_value) {
                sorted_values[n_acceptable] = data[i][f];
                n_acceptable++;
            }
        }
        // Sorting acceptable values
        size_t k;
        data_t x;
        for (int i = 0; i < n_instances; i++) {
            x = sorted_values[i];
            k = i;
            while (k > 0 && sorted_values[k - 1] > x) {
                sorted_values[k] = sorted_values[k - 1];
                k--;
            }
            sorted_values[k] = x;
        }
        // Computing quartiles, deciles, percentiles, etc.
        float step_size = (float) n_acceptable / 100.0;
        size_t current_index = 0;
        int rounded_index = 0;
        for (int i = 0; i < 10; i++) {
            densities[f].deciles[i] = sorted_values[rounded_index];
            for (int k = 0; k < 10; k++) {
                rounded_index = (int) floor(current_index);
                densities[f].percentiles[10 * i + k] = sorted_values[rounded_index];
                current_index += step_size;
            }
        }
        if (use_high_precision && use_high_precision > 100) {
            current_index = 0; rounded_index = 0;
            step_size = (float) use_high_precision / (float) use_high_precision;
            for (int i = 0; i < use_high_precision; i++) {
                rounded_index = (int) floor(current_index);
                densities[f].deciles[i] = sorted_values[rounded_index];
                current_index += step_size;
            }
        }
    }
    return densities;
}

inline double evaluateByThreshold(Splitter* splitter, Density* densities, 
                                 data_t** data, size_t* belongs_to,
                                 int use_high_precision) {
    size_t i = splitter->feature_id;
    size_t best_split_id = 0;
    double lowest_cost = INFINITY;
    double cost = INFINITY;
    // TODO
    return lowest_cost;
}

struct Tree* ID3(data_t** data, data_t* targets, size_t n_instances, size_t n_features,
                 struct Density* densities, struct TreeConfig* config) {
    struct Node* current_node = newNode(config->n_classes);
    struct Tree* tree = (struct Tree*) malloc(sizeof(struct Tree));
    tree->root = *current_node;
    tree->config = config;
    tree->n_nodes = 1;
    tree->n_classes = config->n_classes;
    tree->n_features = n_features;
    size_t* belongs_to = (size_t*) calloc(n_instances, sizeof(size_t));
    struct Splitter splitter = { 
        current_node, 
        n_instances, 
        NO_FEATURE,
        n_features
    };
    bint still_going = TRUE;
    struct Density* next_density;
    size_t best_feature = 0;
    struct Queue* queue = newQueue();
    enqueue(queue, current_node);
    while ((tree->n_nodes < config->max_nodes) && !isQueueEmpty(queue) && still_going) {
        current_node = dequeue(queue);
        double e_cost = INFINITY;
        double lowest_e_cost = INFINITY;
        for (int f = 0; f < n_features; f++) {
            splitter.node = current_node;
            splitter.feature_id = k;
            e_cost = evaluateByThreshold(splitter, densities, data, belongs_to);
            if (e_cost < lowest_e_cost) {
                lowest_e_cost = e_cost;
                best_feature = f;
            }
        }
        if (best_feature != current_node->feature_id) {
            next_density = &densities[best_feature];
            data_t split_value = next_density.split_value;
            current_node.feature_id = best_feature;
            current_node.split_value = split_value;
        }
    }

    free(belongs_to);
    free(queue);
}