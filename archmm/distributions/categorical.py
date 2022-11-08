# -*- coding: utf-8 -*-
#
# categorical.py
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

from archmm.distributions.base import BaseDistribution


class Categorical(BaseDistribution):

    def __init__(self, n_classes: int):
        self.n_classes: int = n_classes
        self.p: np.ndarray = np.random.rand(self.n_classes)
        self.p /= np.sum(self.p)

    def param_update(self, data: np.ndarray, gamma: np.ndarray):
        denominator = np.sum(gamma)
        idx = data.astype(int)
        self.p[:] = 0
        np.add.at(self.p, idx, gamma)
        self.p /= denominator

    def log_pdf(self, data: np.ndarray) -> np.ndarray:
        idx = data.astype(int)
        return np.log(self.p[idx])

    def sample(self, n: int) -> np.ndarray:
        return np.random.choice(np.arange(self.n_classes), size=n, p=self.p)
