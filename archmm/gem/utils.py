# -*- coding: utf-8 -*-
#
# utils.py
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

from typing import Any

import torch


def ensure_tensor(data: Any) -> torch.Tensor:
    if not torch.is_tensor(data):
        data = torch.FloatTensor(data)
    return data


def create_activation_function(name: str) -> torch.nn.Module:
    name = name.strip().lower()
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'sigmoid':
        return torch.nn.Sigmoid()
    elif name == 'tanh':
        return torch.nn.Tanh()
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU()
    else:
        raise NotImplementedError(f'Unknown activation function"{name}"')
