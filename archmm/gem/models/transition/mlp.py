# -*- coding: utf-8 -*-
#
# mlp.py
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

from typing import Tuple, List

import torch

from archmm.gem import TransitionModel
from archmm.gem.utils import create_activation_function


class TransitionMLP(TransitionModel):

    def __init__(self, shape: List[int], activation: str = 'tanh'):
        super().__init__(shape[-1])
        self.shape: Tuple[int] = tuple(shape)
        assert len(self.shape) > 1
        self.layers: torch.nn.Sequential() = torch.nn.Sequential()
        for i in range(len(self.shape) - 1):
            self.layers.append(torch.nn.Linear(self.shape[i], self.shape[i + 1]))
            if i < len(self.shape[-2]):
                self.layers.append(create_activation_function(activation))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layers(X)
        X = torch.nn.functional.log_softmax(X, dim=1)

    def log_prob_(self, X: torch.Tensor) -> torch.Tensor:
        pass
