# -*- coding: utf-8 -*-
#
# beta.py
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

import random

import numpy as np
import torch
from scipy.stats import beta

from archmm.distributions.base import BaseDistribution
from archmm.distributions.support import Bounds


class Beta(BaseDistribution):

    def __init__(self):
        super().__init__()
        self.alpha: float = random.random()
        self.beta: float = random.random()
        self.add_support(Bounds(0, 1))

    def _param_update(self, data: np.ndarray, gamma: np.ndarray):
        raise NotImplementedError('No closed-form ML estimator for Beta distribution')

    def _log_pdf(self, data: np.ndarray) -> np.ndarray:
        return beta.logpdf(data, self.alpha, self.beta)

    def log_pdf_torch(self, data: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Beta(self.alpha, self.beta)
        return dist.log_prob(data)

    def sample(self, n: int) -> np.ndarray:
        return beta.rvs(self.alpha, self.beta, size=n)
