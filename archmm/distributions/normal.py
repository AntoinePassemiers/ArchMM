# -*- coding: utf-8 -*-
#
# normal.py
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


class Gaussian(BaseDistribution):

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features: int = n_features
        self.mu: float = float(np.random.rand())
        self.sigma = float(np.random.rand())
        self.sigma += np.eye(self.n_features)

    def _param_update(self, data: np.ndarray, gamma: np.ndarray):
        denominator = np.sum(gamma)
        self.mu = float(np.sum(data * gamma, axis=0) / denominator)
        centered = data - self.mu
        self.sigma = float(np.sqrt(np.sum((centered ** 2.) * gamma, axis=0) / denominator))

    def _log_pdf(self, data: np.ndarray) -> np.ndarray:
        return scipy.stats.norm.logpdf(data, loc=self.mu, scale=self.sigma)

    def sample(self, n: int) -> np.ndarray:
        return scipy.stats.norm.rvs(loc=self.mu, scale=self.sigma, size=n)
