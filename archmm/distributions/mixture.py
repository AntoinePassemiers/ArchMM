# -*- coding: utf-8 -*-
#
# mixture.py
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

from typing import List

import numpy as np

from archmm.distributions.base import BaseDistribution


class Mixture(BaseDistribution):

    def __init__(self, *components: BaseDistribution):
        super().__init__(*components[0].shape)
        self.components: List[BaseDistribution] = list(components)
        self.weights: np.ndarray = np.random.rand(len(self.components))
        self.weights /= np.sum(self.weights)

    def _param_update(self, data: np.ndarray, gamma: np.ndarray):
        ps = []
        for component, weight in zip(self.components, self.weights):
            ps.append(weight * np.exp(component.log_pdf(data)))
        denominator = np.sum(ps, axis=0)
        for i, component in enumerate(self.components):
            component.param_update(data, gamma * ps[i] / denominator)

    def _log_pdf(self, data: np.ndarray) -> np.ndarray:
        p = np.zeros(len(data), dtype=float)
        for component, weight in zip(self.components, self.weights):
            p += weight * np.exp(component.log_pdf(data))
        return np.log(p)

    def sample(self, n: int) -> np.ndarray:
        data = np.zeros((n, *self.shape), dtype=float)  # TODO: distribution-specific data types
        labels = np.random.choice(np.arange(len(self.weights)), size=n, p=self.weights)
        for i, component in enumerate(self.components):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                data[idx, ...] = component.sample(len(idx))
        return data
