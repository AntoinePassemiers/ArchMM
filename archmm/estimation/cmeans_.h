#ifndef CMEANS__H_
#define CMEANS__H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void fuzzyCMeans(double** X,
                 double** U,
                 double** centroids,
                 int n_features,
                 int n_samples,
                 int n_clusters,
                 int max_n_iter,
                 int fuzzy_coef);


#endif // CMEANS__H_