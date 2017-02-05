#ifndef ID3_H_
#define ID3_H_

#include "queue_.h"
#include "utils_.h"

struct Node;

struct Tree;

inline float ShannonEntropy(float* probabilities, size_t n_classes);

inline float GiniCoefficient(float* probabilities, size_t n_classes);

#endif // ID3_H_