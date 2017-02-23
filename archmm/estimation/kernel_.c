#include "kernel_.h"

inline rbf_distance_t fast_gaussianRBF(data_distance_t r, float epsilon) {
	data_distance_t p = epsilon * r;
	return (rbf_distance_t) exp((double) -p * p);
}

inline rbf_distance_t fast_multiquadricRBF(data_distance_t r, float epsilon) {
	data_distance_t p = epsilon * r;
	return (rbf_distance_t) sqrt((double) 1.0 + (p * p));
}

inline rbf_distance_t fast_inverseQuadraticRBF(data_distance_t r, float epsilon) {
	data_distance_t p = epsilon * r;
	return (rbf_distance_t) 1.0 / (1.0 + (p * p));
}

inline rbf_distance_t fast_inverseMultiquadricRBF(data_distance_t r, float epsilon) {
	data_distance_t p = epsilon * r;
	return (rbf_distance_t) 1.0 / sqrt((double) (1.0 + (p * p)));
}

inline rbf_distance_t fast_polyharmonicSplineRBF(data_distance_t r, double k) {
	return (rbf_distance_t) pow((double) r, k);
}

inline rbf_distance_t fast_thinPlateSplineRBF(data_distance_t r, double k) {
	return (rbf_distance_t) pow((double) r, 2) * log(r);
}