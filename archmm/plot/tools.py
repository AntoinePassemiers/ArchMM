# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

from archmm.estimation.clustering import *
from archmm.utils import *

means = np.array([[-23, 78], [51, -36], [19, 7]])
covs1 = np.array([[1, 0], [-4.8, 5]]) * 70.0
covs2 = np.array([[0.70, -0.50], [1.40, 3.2]]) * 80.0
covs3 = np.array([[2.0, 0.80], [-3.30, 0.9]]) * 80.0
X1 = np.random.multivariate_normal(means[0], covs1, size = 500)
X2 = np.random.multivariate_normal(means[1], covs2, size = 500)
X3 = np.random.multivariate_normal(means[2], covs3, size = 500)
X = np.concatenate((X1, X2, X3), axis = 0)

kmeans, labels = kMeans(X, 3, n_iter = 100)

colors = ["r", "g", "b"]

for i in range(3):
	points = X[labels == i]
	plt.scatter(points[:, 0], points[:, 1], color = colors[i])
	plt.scatter([kmeans[i, 0]], [kmeans[i, 1]], color = "black", s = 100)
plt.show()

cmeans, U = fuzzyCMeans(X, 3)

for i in range(len(X)):
	plt.scatter([X[i, 0]], [X[i, 1]], color = U[i])
plt.show()