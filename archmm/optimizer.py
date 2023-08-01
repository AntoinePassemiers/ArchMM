# -*- coding: utf-8 -*-
#
# optimizer.py
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

from typing import Optional, List, Tuple

import numpy as np
import torch.optim
from archmm.parameter import BaseParameter

from archmm.state import HiddenState


class Optimizer:

    def __init__(
            self,
            n_iter_per_em_step: int = 10,
            lr: float = 0.0002
    ):
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.states: List[HiddenState] = []
        self.priors: List[Tuple[BaseParameter, torch.distributions.Distribution]] = []
        self.n_iter_per_em_step: int = n_iter_per_em_step
        self.lr: float = lr

    def init(self, states: List[HiddenState]):
        self.states = []
        parameters = []
        for state in states:
            self.states.append(state)
        for state in states:
            parameters += list(state.dist.parameters())
        if len(parameters) > 0:
            self._optimizer = torch.optim.Adam(parameters, lr=self.lr)

    def add_prior(self, param: BaseParameter, dist: torch.distributions.Distribution):
        self.priors.append((param, dist))

    def update_params(self, sequence: np.ndarray, gamma: np.ndarray):

        # Update parameters for which closed-form ML estimators exist
        for i in range(len(self.states)):
            if self.states[i].dist.closed_form:
                self.states[i].dist.param_update(sequence, gamma[:, i])

        # Gradient-based iterative optimization of parameters for which
        # closed-form ML estimators don't exist
        for _ in range(self.n_iter_per_em_step):
            if self._optimizer is not None:
                self._optimizer.zero_grad()
            loss = torch.FloatTensor(np.asarray([0.]))
            for i in range(len(self.states)):
                if not self.states[i].dist.closed_form:
                    loss = loss - torch.sum(
                        torch.FloatTensor(gamma[:, i]) * self.states[i].dist.log_pdf_torch(
                            torch.FloatTensor(sequence)))

            # Priors
            for param, dist in self.priors:
                loss = loss - torch.sum(dist.log_prob(param.get()))

            if loss.requires_grad:
                loss.backward()
                if self._optimizer is not None:
                    self._optimizer.step()
