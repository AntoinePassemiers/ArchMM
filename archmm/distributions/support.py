# -*- coding: utf-8 -*-
#
# support.py
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
from typing import Tuple

import numpy as np


class Support(metaclass=ABCMeta):

    class InvalidData(Exception):
        pass

    def __call__(self, data: np.ndarray):
        if not self.check(data):
            raise Support.InvalidData(f'Invalid data: constraint "{self.__str__()}" not satisfied')

    @abstractmethod
    def check(self, data: np.ndarray) -> bool:
        pass


class Integer(Support):

    def check(self, data: np.ndarray) -> bool:
        return data.dtype.kind == 'i'

    def __str__(self) -> str:
        return 'integer'


class RealValued(Support):

    def check(self, data: np.ndarray) -> bool:
        return data.dtype.kind == 'f'

    def __str__(self) -> str:
        return 'real-valued'


class Positive(Support):

    def check(self, data: np.ndarray) -> bool:
        return np.all(np.positive(data))

    def __str__(self) -> str:
        return '0 <= value'


class Bounds(Support):

    def __init__(self, lb: float = -float('inf'), ub: float = float('inf')):
        self.lb: float = float(lb)
        self.ub: float = float(ub)

    def check(self, data: np.ndarray) -> bool:
        return np.all(data >= self.lb) and (np.all(data <= self.ub))

    def __str__(self) -> str:
        return f'{self.lb} <= value <= {self.ub}'


class Shape(Support):

    def __init__(self, *shape: int):
        self.shape: Tuple[int] = tuple(shape)

    def check(self, data: np.ndarray) -> bool:
        return tuple(data.shape[1:]) == self.shape

    def __str__(self) -> str:
        return f'len(data.shape) == {self.shape}'
