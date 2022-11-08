# -*- coding: utf-8 -*-
#
# gaussian.py
#
# Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.

import numpy as np
import scipy.stats

from archmm.distributions.base import BaseDistribution


class MultivariateGaussian(BaseDistribution):

    def __init__(self, n_features: int):
        self.n_features: int = n_features
        self.mu: np.ndarray = np.random.rand(self.n_features)
        self.sigma = np.random.rand(self.n_features, self.n_features)
        self.sigma += np.eye(self.n_features)

    def param_update(self, data: np.ndarray, gamma: np.ndarray):
        denominator = np.sum(gamma)
        self.mu[:] = np.sum(data * gamma[:, np.newaxis], axis=0) / denominator
        centered = data - self.mu[np.newaxis, :]
        self.sigma[:, :] = np.einsum('t,tk,tl->kl', gamma, centered, centered) / denominator

    def log_pdf(self, data: np.ndarray) -> np.ndarray:
        return scipy.stats.multivariate_normal.logpdf(data, mean=self.mu, cov=self.sigma)

    def sample(self, n: int) -> np.ndarray:
        return scipy.stats.multivariate_normal.rvs(mean=self.mu, cov=self.sigma, size=n)
