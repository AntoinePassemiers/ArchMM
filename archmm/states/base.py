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

import numpy as np


class HiddenState(metaclass=ABCMeta):

    def __init__(self, n_features: int):
        self.n_features: int = n_features

    @abstractmethod
    def param_update(self, data: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def log_pdf(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        pass
