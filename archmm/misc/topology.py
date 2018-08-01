# -*- coding: utf-8 -*-
# topology.py
# author: Antoine Passemiers

import numpy as np


class Topology:

    def __init__(self, n_states):
        self.n_states = n_states
        self.transition_mask = np.zeros(
            (self.n_states, self.n_states), dtype=np.bool)
    
    def add_self_loop(self, src):
        self.transition_mask[src, src] = True

    def add_self_loops(self):
        for src in range(self.n_states):
            self.add_self_loop(src)

    def add_edge(self, src, dest):
        self.transition_mask[src, dest] = True
    
    def add_edges(self, func):
        for src in range(self.n_states):
            dest = func(src)
            if 0 <= dest < self.n_states:
                self.add_edge(src, dest)
    
    def add_edges_everywhere(self):
        for src in range(self.n_states):
            for dest in range(self.n_states):
                self.add_edge(src, dest)
    
    def remove_edge(self, src, dest):
        self.transition_mask[src, dest] = False
    
    def to_mask(self):
        return np.asarray(self.transition_mask, dtype=np.int)