# -*- coding: utf-8 -*-
#
# state.py
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

from typing import List

from archmm.distributions.base import BaseDistribution


class HiddenState:

    def __init__(self, dist: BaseDistribution):
        self.dist: BaseDistribution = dist
        self.transitions: List[HiddenState] = []
        self.can_start: bool = False

    def allow_start(self, can_start: bool = True):
        self.can_start = bool(can_start)

    def add_transition(self, state: 'HiddenState'):
        self.transitions.append(state)

    def remove_transitions(self):
        self.transitions = []
        self.can_start = False

    def can_transit_to(self, other: 'HiddenState') -> bool:
        return other in self.transitions

    def is_allowed_to_start(self) -> bool:
        return self.can_start


class Architecture:

    @staticmethod
    def ergodic(states: List['HiddenState']):
        for i in range(len(states)):
            states[i].remove_transitions()
            states[i].allow_start()
            for j in range(len(states)):
                states[i].add_transition(states[j])

    @staticmethod
    def linear(states: List['HiddenState']):
        for i in range(len(states) - 1):
            states[i].remove_transitions()
            states[i].allow_start(can_start=(i == 0))
            states[i].add_transition(states[i])
            states[i].add_transition(states[i + 1])
        if len(states) > 0:
            states[-1].remove_transitions()
            states[-1].add_transition(states[-1])

    @staticmethod
    def cyclic(states: List['HiddenState']):
        for i in range(len(states)):
            states[i].remove_transitions()
            j = (i + 1) % len(states)
            states[i].add_transition(states[i])
            states[i].add_transition(states[j])
            states[i].allow_start()
