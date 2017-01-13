# -*- coding: utf-8 -*-

from threading import Thread

class TrainingThread(Thread):
    def __init__(self, group = None, target = None, args = None, kwargs = None,
                 name = None, verbose = None):
        Thread.__init__(self, group, target, name, args, kwargs, verbose)
        self.results = None
        
    def run(self):
        if self._Thread__target is not None:
            self.results = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            
    def join(self, *args, **kwargs):
        Thread.join(self, *args, **kwargs)
        return self.results
    
def train_pi_network(network, U, gamma, weights, **kwargs):
    return TrainingThread(target = network.train, 
                          args = (U, gamma, weights,), kwargs = kwargs)

def train_state_network(network, U, xi, weights, is_mv, **kwargs):
    return TrainingThread(target = network.train, 
                          args = (U, xi, weights, is_mv,), kwargs = kwargs)

def train_output_network(network, U, targets, memory, weights, is_mv, **kwargs):
    return TrainingThread(target = network.train, 
                          args = (U, targets, memory, weights, is_mv,), kwargs = kwargs)