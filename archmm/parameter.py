# -*- coding: utf-8 -*-
#
# parameter.py
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

import torch


class BaseParameter(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, shape: tuple):
        super().__init__()
        self.raw: torch.nn.Parameter = torch.nn.Parameter(torch.randn(*shape))

    @abstractmethod
    def get(self) -> torch.Tensor:
        pass

    @abstractmethod
    def set(self, x: torch.Tensor):
        pass


class Parameter(BaseParameter):

    def __init__(self, shape: tuple):
        super().__init__(shape)

    def get(self) -> torch.Tensor:
        return self.raw

    def set(self, x: torch.Tensor):
        self.raw.data = x


class DoublyBoundedParameter(BaseParameter):

    def __init__(self, shape: tuple, lb: float, ub: float):
        super().__init__(shape)
        self.lb: float = lb
        self.ub: float = ub

    def get(self) -> torch.Tensor:
        x = torch.sigmoid(self.raw)
        return (self.ub - self.lb) * x + self.lb

    def set(self, x: torch.Tensor):
        x = (x - self.lb) / (self.ub - self.lb)
        self.raw.data = torch.logit(x)


class PositiveParameter(BaseParameter):

    def __init__(self, shape: tuple):
        super().__init__(shape)

    def get(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.raw, -100, 100))

    def set(self, x: torch.Tensor):
        self.raw.data = torch.log(x)
