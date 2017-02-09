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

inline float ShannonEntropy(float probability) {
    return -probability * log2(probability);
}

inline float GiniCoefficient(float probability) {
    return 1.0 - probability * probability;
}

struct Density* computeDensities(data_t* data, size_t n_instances, size_t n_features, 
                                 size_t n_classes, int use_high_precision,
                                 data_t nan_value) {
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
        if (use_high_precision) {
            densities[f].high_precision_partition = (data_t*) malloc((use_high_precision - 1) * s);
        }
        else {
            densities[f].high_precision_partition = NULL;
        }
        // Putting nan values aside
        size_t n_acceptable = 0;
        for (int i = 0; i < n_instances; i++) {
            if (data[i * n_features + f] != nan_value) {
                sorted_values[n_acceptable] = data[i * n_features + f];
                n_acceptable++;
            }
        }
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
        // Computing quartiles, deciles, percentiles, etc.
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
        if (use_high_precision > 100) {
            current_index = 0; rounded_index = 0;
            step_size = (float) (n_acceptable - 1) / (float) use_high_precision;
            for (int i = 0; i < use_high_precision; i++) {
                rounded_index = (int) floor(current_index);
                densities[f].deciles[i] = sorted_values[rounded_index];
                current_index += step_size;
            }
        }
    }
    free(sorted_values);
    return densities;
}

inline double getFeatureCost(struct Density* density, size_t n_classes) {
    size_t n_left = sum_counts(density->counters_left, n_classes);
    size_t n_right = sum_counts(density->counters_right, n_classes);
    if (n_left == 0 || n_right == 0) {
        return COST_OF_EMPTINESS;
    }
    size_t total = n_left + n_right;
    double cost = 0.0;
    for (int i = 0; i < n_classes; i++) {
        size_t n_p = density->counters_left[i];
        size_t n_n = density->counters_right[i];
        if (n_left > 0 && n_p > 0) {
            cost += ((float) n_left / total) * ShannonEntropy((float) n_p / n_left);
        }
        if (n_right > 0 && n_n > 0) {
            cost += ((float) n_right / total) * ShannonEntropy((float) n_n / n_right);
        }
    }
    return cost;
}

double evaluatePartitions(data_t* data, struct Density* density,
                          data_t* partition_values, struct Splitter* splitter, 
                          size_t n_classes, size_t* belongs_to, size_t k) {
    size_t i = splitter->feature_id;
    size_t n_features = splitter->n_features;
    data_t data_point;
    target_t target_value;
    memset((void*) density->counters_left, 0x00, n_classes * sizeof(size_t));
    memset((void*) density->counters_right, 0x00, n_classes * sizeof(size_t));
    memset((void*) density->counters_nan, 0x00, n_classes * sizeof(size_t));
    density->split_value = partition_values[k];
    #if ID3_DEBUG_MODE && DEBUG_LEVEL >= 4
        printf("\tDensity split value : %f\n", density->split_value);
    #endif
    for (int j = 0; j < splitter->n_instances; j++) {
        if (belongs_to[j] == splitter->node->id) {
            target_value = splitter->targets[j];
            if (!(0 <= target_value && target_value < n_classes)) {
                printf("Error : one of the target values is not in [0, n_classes].");
                exit(EXIT_FAILURE);
            }
            data_point = data[j * n_features + i];
            if (data_point == splitter->nan_value) {
                density->counters_nan[target_value]++;
            }
            else if (data_point >= density->split_value) {
                density->counters_right[target_value]++;
            }
            else {
                density->counters_left[target_value]++;
            }
        }
    }
    return getFeatureCost(density, n_classes);
}

double evaluateByThreshold(struct Splitter* splitter, struct Density* density, 
                           data_t* data, size_t* belongs_to, size_t n_classes,
                           int partition_value_type, size_t n_partition_values) {
    size_t best_split_id = 0;
    double lowest_cost = INFINITY;
    double cost;
    data_t* partition_values;
    switch(partition_value_type) {
        case QUARTILE_PARTITIONING:
            partition_values = density->quartiles;
            break;
        case DECILE_PARTITIONING:
            partition_values = density->deciles;
            break;
        case PERCENTILE_PARTITIONING:
            partition_values = density->percentiles;
            break;
        case HIGH_PRECISION_PARTITIONING:
            partition_values = density->high_precision_partition;
        default:
            partition_values = density->quartiles;
    }
    for (int k = 0; k < n_partition_values; k++) {
        cost = evaluatePartitions(data, density, partition_values, splitter, n_classes, 
                                  belongs_to, k);
        if (cost < lowest_cost && cost != COST_OF_EMPTINESS) {
            lowest_cost = cost;
            best_split_id = k;
        }
    }
    // density->split_value = partition_values[best_split_id];
    evaluatePartitions(data, density, partition_values, splitter, 
                       n_classes, belongs_to, best_split_id);
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
        NO_FEATURE,
        n_features,
        targets,
        config->nan_value
    };
    struct Density* densities = computeDensities(data, n_instances, n_features, 
        config->n_classes, config->use_high_precision, config->nan_value);
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
            e_cost = evaluateByThreshold(&splitter, &densities[f], data, belongs_to,
                                         config->n_classes, PERCENTILE_PARTITIONING, 100); // TODO
            if (e_cost < lowest_e_cost) {
                lowest_e_cost = e_cost;
                best_feature = f;
            }
        }
        next_density = &densities[best_feature];
        #if ID3_DEBUG_MODE && DEBUG_LEVEL >= 1
            printf("-------------------\n");
            printf("Node number %i\n", tree->n_nodes);
            printf("Best feature : %i\n", (int) best_feature);
            printf("Split value : %f\n", (double) next_density->split_value);
            printf("Split cost : %f\n", (double) lowest_e_cost);
            printf("%i, %i\n", best_feature, current_node->feature_id);
        #endif
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
                    child_node->counters = (size_t*) malloc(config->n_classes * sizeof(size_t));
                    child_node->n_instances = split_totals[i];
                    child_node->score = COST_OF_EMPTINESS;
                    child_node->feature_id = best_feature;
                    child_node->left_child = NULL;
                    child_node->right_child = NULL;
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