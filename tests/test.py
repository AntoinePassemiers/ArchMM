# -*- coding: utf-8 -*-
#
# test.py
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

from archmm.hmm import HMM
from archmm.distributions import MultivariateGaussian


def test_gaussian():
    sequences = []
    sequence = np.random.rand(1800, 3)
    sequence[1200:, :] += 0.5
    sequences.append(sequence)
    sequence = np.random.rand(1800, 3)
    sequence[300:, :] += 0.5
    sequences.append(sequence)

    model = HMM()
    for _ in range(3):
        model.add_state(MultivariateGaussian(3))
    model.fit(sequences)

    for sequence in sequences:
        model.decode(sequence)
