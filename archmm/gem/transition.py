# -*- coding: utf-8 -*-
#
# transition.py
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
from typing import Any

import torch

from archmm.gem.utils import ensure_tensor


class TransitionModel(metaclass=ABCMeta):

    def __init__(self, n_states: int):
        self.n_states: int = n_states

    @abstractmethod
    def log_prob_(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def log_prob(self, X: Any) -> torch.Tensor:
        X = ensure_tensor(X)
        p = self.log_prob_(X)
        assert len(p.size()) == 2
        assert p.size()[-1] == self.n_states
        return p
