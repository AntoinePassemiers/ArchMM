#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "id3_.h"


void ID4_Update(struct Tree* tree, data_t* data, target_t* targets, size_t n_instances, 
		        size_t n_features, size_t n_classes, data_t nan_value) {
	struct Node* current_node;
    for (int i = 0; i < n_instances; i++) {
    	current_node = tree->root;
    	target_t target = targets[i];
    	bint still_going = TRUE;
    	while (still_going) {
    		size_t feature_id = current_node->feature_id;
    		current_node->n_instances++;
    		current_node->counters[target]++;
    		if ((data[i * n_features + feature_id] == nan_value) || 
    			(current_node->left_child == NULL) || (current_node->right_child == NULL)) {
    			still_going = FALSE;
    		}
    		else {
    			size_t feature_id = current_node->feature_id;
    			if (data[i * n_features + feature_id] < current_node->split_value) {
    				current_node = current_node->left_child;
    			}
    			else {
    				current_node = current_node->right_child;
    			}
    		}
    	}
    }
}