#include <math.h>  

#include "utils_.h"


inline float log_2(float value) {  
    return log(value) / log(2);
}

inline size_t sum_counts(size_t* counters, size_t n_counters) {
	size_t total = 0;
	for (int i = 0; i < n_counters; i++) {
		total += counters[i];
	}
	return total;
}