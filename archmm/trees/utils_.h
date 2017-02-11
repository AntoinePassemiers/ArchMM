#ifndef UTILS__H_
#define UTILS__H_

#include <stddef.h>

#define ID3_DEBUG_MODE 0
#define DEBUG_LEVEL 4

#define TRUE 1
#define FALSE 0

typedef int bint;
typedef double data_t;
typedef int target_t;

inline float log_2(float value);

extern inline size_t sum_counts(size_t* counters, size_t n_counters);

#endif // UTILS__H_