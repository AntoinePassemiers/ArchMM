#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

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

struct Density* computeDensities(data_t* data, size_t n_instances, size_t n_features, 
                                 size_t n_classes, data_t nan_value) {
    size_t s = sizeof(data_t);
    struct Density* densities = (struct Density*) malloc(n_features * sizeof(struct Density));
    data_t* sorted_values = (data_t*) malloc(n_instances * s);
    for (int f = 0; f < n_features; f++) {
        densities[f].quartiles = (data_t*) malloc(4 * s);
        densities[f].deciles = (data_t*) malloc(10 * s);
        densities[f].percentiles = (data_t*) malloc(100 * s);
        densities[f].counters_left = (size_t*) malloc(n_classes * sizeof(size_t));
        densities[f].counters_right = (size_t*) malloc(n_classes * sizeof(size_t));
        densities[f].counters_nan = (size_t*) malloc(n_classes * sizeof(size_t));
        // Putting nan values aside
        bint is_categorical = TRUE;
        size_t n_acceptable = 0;
        data_t data_point;
        for (int i = 0; i < n_instances; i++) {
            data_point = data[i * n_features + f];
            if (data_point != nan_value) {
                sorted_values[n_acceptable] = data_point;
                n_acceptable++;
                if (is_categorical && !(round(data_point) == data_point)) {
                    is_categorical = FALSE;
                }
            }
        }
        densities[f].is_categorical = is_categorical;
        // Sorting acceptable values
        size_t k;
        data_t x;
        for (int i = 0; i < n_acceptable; i++) {
            x = sorted_values[i];
            k = i;
            while (k > 0 && sorted_values[k - 1] > x) {
                sorted_values[k] = sorted_values[k - 1];
                k--;
            }
            sorted_values[k] = x;
        }
        // Computing quartiles, deciles, percentiles
        float step_size = (float) n_acceptable / 100.0;
        float current_index = 0.0;
        int rounded_index = 0;
        for (int i = 0; i < 10; i++) {
            densities[f].deciles[i] = sorted_values[rounded_index];
            for (int k = 0; k < 10; k++) {
                rounded_index = (int) floor(current_index);
                densities[f].percentiles[10 * i + k] = sorted_values[rounded_index];
                current_index += step_size;
            }
        }
    }
    free(sorted_values);
    return densities;
}

inline float ShannonEntropy(float probability) {
    return -probability * log2(probability);
}

inline float GiniCoefficient(float probability) {
    return 1.0 - probability * probability;
}

inline double getFeatureCost(struct Density* density, size_t n_classes) {
    size_t n_left = sum_counts(density->counters_left, n_classes);
    size_t n_right = sum_counts(density->counters_right, n_classes);
    size_t total = n_left + n_right;
    float left_rate = (float) n_left / total;
    float right_rate = (float) n_right / total;
    if (n_left == 0 || n_right == 0) {
        return COST_OF_EMPTINESS;
    }
    double left_cost = 0.0, right_cost = 0.0;
    size_t* counters_left = density->counters_left;
    size_t* counters_right = density->counters_right;
    if (n_left > 0) {
        size_t n_p;
        for (int i = 0; i < n_classes; i++) {
            n_p = counters_left[i];
            if (n_p > 0) {
                left_cost += ShannonEntropy((float) n_p / n_left);
            }
        }
        left_cost *= left_rate;
    }
    if (n_right > 0) {
        size_t n_n;
        for (int i = 0; i < n_classes; i++) {
            n_n = counters_right[i];
            if (n_n > 0) {
                right_cost += ShannonEntropy((float) n_n / n_right);
            }
        }
        right_cost *= right_rate;
    }
    return left_cost + right_cost;
}

inline double evaluatePartitions(data_t* data, struct Density* density,
                                 struct Splitter* splitter, size_t k) {
    size_t i = splitter->feature_id;
    size_t n_features = splitter->n_features;
    data_t data_point;
    target_t target_value;
    size_t id = splitter->node->id;
    memset((void*) density->counters_left, 0x00, splitter->n_classes * sizeof(size_t));
    memset((void*) density->counters_right, 0x00, splitter->n_classes * sizeof(size_t));
    memset((void*) density->counters_nan, 0x00, splitter->n_classes * sizeof(size_t));
    density->split_value = splitter->partition_values[k];
    data_t split_value = density->split_value;
    for (int j = 0; j < splitter->n_instances; j++) {
        if (splitter->belongs_to[j] == id) {
            target_value = splitter->targets[j];
            data_point = data[j * n_features + i];
            if (data_point == splitter->nan_value) {
                density->counters_nan[target_value]++;
            }
            else if (data_point >= split_value) {
                density->counters_right[target_value]++;
            }
            else {
                density->counters_left[target_value]++;
            }
        }
    }
    return getFeatureCost(density, splitter->n_classes);
}

double evaluateByThreshold(struct Splitter* splitter, struct Density* density, 
                           data_t* data, int partition_value_type) {
    size_t best_split_id = 0;
    double lowest_cost = INFINITY;
    double cost;
    size_t n_partition_values;
    switch(partition_value_type) {
        case QUARTILE_PARTITIONING:
            splitter->partition_values = density->quartiles;
            n_partition_values = 4;
            break;
        case DECILE_PARTITIONING:
            splitter->partition_values = density->deciles;
            n_partition_values = 10;
            break;
        case PERCENTILE_PARTITIONING:
            splitter->partition_values = density->percentiles;
            n_partition_values = 100;
            break;
        default:
            splitter->partition_values = density->percentiles;
            n_partition_values = 100;
    }
    for (int k = 1; k < n_partition_values - 1; k++) {
        cost = evaluatePartitions(data, density, splitter, k);
        if (cost < lowest_cost) {
            lowest_cost = cost;
            best_split_id = k;
        }
    }
    evaluatePartitions(data, density, splitter, best_split_id);
    return lowest_cost;
}

void initRoot(struct Node* root, target_t* targets, size_t n_instances, size_t n_classes) {
    memset(root->counters, 0x00, n_classes * sizeof(size_t));
    for (int i = 0; i < n_instances; i++) {
        root->counters[targets[i]]++;
    }
}

struct Tree* ID3(data_t* data, target_t* targets, size_t n_instances, size_t n_features,
                 struct TreeConfig* config) {
    struct Node* current_node = newNode(config->n_classes);
    current_node->id = 0;
    current_node->n_instances = n_instances;
    current_node->score = 0.0;
    initRoot(current_node, targets, n_instances, config->n_classes);
    struct Node* child_node;
    struct Tree* tree = (struct Tree*) malloc(sizeof(struct Tree));
    tree->root = current_node;
    tree->config = config;
    tree->n_nodes = 1;
    tree->n_classes = config->n_classes;
    tree->n_features = n_features;
    bint still_going = 1;
    size_t* belongs_to = (size_t*) calloc(n_instances, sizeof(size_t));
    size_t** split_sides = (size_t**) malloc(2 * sizeof(size_t*));
    struct Splitter splitter = { 
        current_node, 
        n_instances,
        NULL,
        config->n_classes,
        belongs_to,
        NO_FEATURE,
        n_features,
        targets,
        config->nan_value
    };
    struct Density* densities = computeDensities(data, n_instances, n_features, 
        config->n_classes, config->nan_value);
    struct Density* next_density;
    size_t best_feature = 0;
    struct Queue* queue = newQueue();
    enqueue(queue, current_node);
    while ((tree->n_nodes < config->max_nodes) && !isQueueEmpty(queue) && still_going) {
        current_node = dequeue(queue);
        double e_cost = INFINITY;
        double lowest_e_cost = INFINITY;
        splitter.node = current_node;
        for (int f = 0; f < n_features; f++) {
            splitter.feature_id = f;
            e_cost = evaluateByThreshold(&splitter, &densities[f], data, config->partitioning);
            if (e_cost < lowest_e_cost) {
                lowest_e_cost = e_cost;
                best_feature = f;
            }
        }
        next_density = &densities[best_feature];
        if ((best_feature != current_node->feature_id)
            || (next_density->split_value != current_node->split_value)) { // TO REMOVE ?
            next_density = &densities[best_feature];
            size_t split_totals[2] = {
                sum_counts(next_density->counters_left, config->n_classes),
                sum_counts(next_density->counters_right, config->n_classes)
            };
            if (split_totals[0] && split_totals[1]) {
                struct Node* new_children = (struct Node*) malloc(2 * sizeof(struct Node));
                size_t split_value = next_density->split_value;
                current_node->score = lowest_e_cost;
                current_node->feature_id = best_feature;
                current_node->split_value = split_value;
                current_node->left_child = &new_children[0];
                current_node->right_child = &new_children[1];
                split_sides[0] = next_density->counters_left;
                split_sides[1] = next_density->counters_right;
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < n_instances; j++) {
                        bint is_on_the_left = (data[j * n_features + best_feature] < split_value) ? 1 : 0;
                        if (belongs_to[j] == current_node->id) {
                            if (is_on_the_left * (1 - i) + (1 - is_on_the_left) * i) {
                                belongs_to[j] = tree->n_nodes;
                            }
                        }
                    }
                    child_node = &new_children[i];
                    child_node->id = tree->n_nodes;
                    child_node->split_value = split_value;
                    child_node->n_instances = split_totals[i];
                    child_node->score = COST_OF_EMPTINESS;
                    child_node->feature_id = best_feature;
                    child_node->left_child = NULL;
                    child_node->right_child = NULL;
                    child_node->counters = (size_t*) malloc(config->n_classes * sizeof(size_t));
                    memcpy(child_node->counters, split_sides[i], config->n_classes * sizeof(size_t));
                    if (lowest_e_cost > config->min_threshold) {
                        enqueue(queue, child_node);
                    }
                    ++tree->n_nodes;
                }
            }
        }
    }
    free(belongs_to);
    free(queue);
    free(split_sides);
    return tree;
}

float* classify(data_t* data, size_t n_instances, size_t n_features,
                   struct Tree* tree, struct TreeConfig* config) {
    struct Node *current_node, *temp_node;
    size_t feature;
    size_t n_classes = config->n_classes;
    float* predictions = (float*) malloc(n_instances * n_classes * sizeof(float));
    for (int k = 0; k < n_instances; k++) {
        bint improving = TRUE;
        current_node = tree->root;
        while (improving) {
            feature = current_node->feature_id;
            if (current_node->left_child != NULL) {
                if (data[k * n_features + feature] >= current_node->split_value) {
                    current_node = current_node->right_child;
                }
                else {
                    current_node = current_node->left_child;
                }
            }
            else {
                improving = FALSE;
            }
        }
        size_t node_instances = current_node->n_instances;
        for (int c = 0; c < n_classes; c++) {
            predictions[k * n_classes + c] = (float) current_node->counters[c] / node_instances;
        }
    }
    return predictions;
}