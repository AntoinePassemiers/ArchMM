# -*- coding: utf-8 -*-
# Prototyping Markov random fields

import numpy as np
from scipy.misc import imread
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from archmm.mrf import MarkovRandomField


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

n_classes = 9
targets = [imread("Segmentation/%i.png" % i) for i in range(n_classes)]
X = [rgb2gray(x) for x in targets]
# X = [x[:, :, :3] for x in X]
Y = [np.full(x.shape[:2], i, dtype = np.int) for i, x in enumerate(X)]


mrf = MarkovRandomField(n_classes)
mrf.fit(X, Y)
img = imread("Segmentation/1840_paris_crop.png")


img = np.asarray(rgb2gray(img), dtype = np.uint8)
omega = mrf.simulated_annealing(img, max_n_iter = 100)

cmap = ListedColormap(["red", "white", "orange", "yellow",
    "black", "green", "blue", "cyan", "purple"], 'indexed')
plt.imshow(omega, cmap = cmap)
plt.show()

print("Finished")
