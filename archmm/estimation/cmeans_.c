#include "cmeans_.h"


void fuzzyCMeans(double** X,
                 double** U,
                 double** centroids,
                 int n_features,
                 int n_samples,
                 int n_clusters,
                 int max_n_iter,
                 int fuzzy_coef) {
    int n_iter = 0;
    while (n_iter++ < max_n_iter) {
        int i, j, k, l;
        for (j = 0; j < n_clusters; j++) {
            memset((void*) centroids[j], 0x00, n_features * sizeof(double));
            double denom = 0.0;            
            for (i = 0; i < n_samples; i++) {
                for (l = 0; l < n_features; l++) {
                    centroids[j][l] += U[i][j] * X[i][l];
                }
                denom += U[i][j];
            }
            for (l = 0; l < n_features; l++) {
                centroids[j][l] /= denom;
            }
        }
        int n_unchanged = 0;
        for (j = 0; j < n_clusters; j++) {
            for (i = 0; i < n_samples; i++) {
                double denomnum = 0.0;
                for (l = 0; l < n_features; l++) {
                    denomnum += pow(X[i][l] - centroids[j][l], 2);
                }
                denomnum = sqrt(denomnum);
                double denom = 0.0;
                for (k = 0; k < n_clusters; k++) {
                    double denomdenom = 0.0;
                    for (l = 0; l < n_features; l++) {
                        denomdenom += pow(X[i][l] - centroids[k][l], 2);
                    }
                    denomdenom = sqrt(denomdenom);
                    denom += pow(
                        denomnum / denomdenom,
                        2.0 / (fuzzy_coef - 1.0));
                }
                double temp = 1.0 / denom;
                if (temp == U[i][j]) {
                    ++n_unchanged;
                } else {
                    U[i][j] = temp;                    
                }
            }
        }
        if (n_unchanged == n_clusters * n_samples) {
            // System has converged
            n_iter = max_n_iter;
        }
    }
}
