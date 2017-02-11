#ifndef ID4__H_
#define ID4__H_

#include "queue_.h"
#include <stddef.h>


void ID4_Update(struct Tree* tree, const data_t* data, const target_t* targets, 
                size_t n_instances, size_t n_features, size_t n_classes, data_t nan_value);

#endif // ID4__H_