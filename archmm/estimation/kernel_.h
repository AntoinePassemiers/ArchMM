#ifndef KERNEL__H_
#define KERNEL__H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>


typedef double rbf_distance_t;
typedef double data_distance_t;

inline rbf_distance_t fast_gaussianRBF(data_distance_t r, float epsilon);
inline rbf_distance_t fast_multiquadricRBF(data_distance_t r, float epsilon);
inline rbf_distance_t fast_inverseQuadraticRBF(data_distance_t r, float epsilon);
inline rbf_distance_t fast_inverseMultiquadricRBF(data_distance_t r, float epsilon);
inline rbf_distance_t fast_polyharmonicSplineRBF(data_distance_t r, double k);
inline rbf_distance_t fast_thinPlateSplineRBF(data_distance_t r, double k);


#endif // KERNEL__H_