# -*- coding: utf-8 -*-
#
# base.py
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

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np
import torch

from archmm.distributions.support import Support, Shape


class BaseDistribution(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, *shape: int, closed_form: bool = True):
        super().__init__()
        self.shape: Tuple[int] = tuple(shape)
        self.closed_form: bool = closed_form
        self.supports: List[Support] = []
        self.add_support(Shape(*self.shape))

    def add_support(self, support: Support):
        self.supports.append(support)

    def check_data(self, data: np.ndarray):
        for support in self.supports:
            support(data)

    def param_update(self, data: np.ndarray, gamma: np.ndarray):
        self.check_data(data)
        self._param_update(data, gamma)

    @abstractmethod
    def _param_update(self, data: np.ndarray, gamma: np.ndarray):
        pass

    def log_pdf(self, data: np.ndarray) -> np.ndarray:
        self.check_data(data)
        return self._log_pdf(data)

    @abstractmethod
    def _log_pdf(self, data: np.ndarray) -> np.ndarray:
        pass

    def log_pdf_torch(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        pass
