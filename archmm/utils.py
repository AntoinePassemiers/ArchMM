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

from typing import List, Tuple, Union

import numpy as np


def check_data(data: Union[np.ndarray, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(data, list):
        if len(data) == 0:
            return np.array([]), np.array([0])
        bounds = []
        start = 0
        for seq in data:
            bounds.append(start)
            start += len(seq)
        bounds.append(start)
        data = np.concatenate(data, axis=0)
    else:
        bounds = [0, len(data)]
    return data, np.asarray(bounds, dtype=int)
