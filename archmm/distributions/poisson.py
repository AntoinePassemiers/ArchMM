# -*- coding: utf-8 -*-
#
# poisson.py
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
from archmm.distributions.support import Integer, Positive


class Poisson(BaseDistribution):

    def __init__(self):
        super().__init__()
        self.lambda_: float = float(np.random.randint(1, 20))
        self.add_support(Integer())
        self.add_support(Positive())

    def _param_update(self, data: np.ndarray, gamma: np.ndarray):
        denominator = np.sum(gamma)
        self.lambda_ = np.sum(data * gamma)
        self.lambda_ /= denominator

    def _log_pdf(self, data: np.ndarray) -> np.ndarray:
        return scipy.stats.poisson.logpmf(data, self.lambda_)

    def sample(self, n: int) -> np.ndarray:
        return np.random.poisson(lam=self.lambda_, size=n)
