""" A fixed point solver
"""
from abc import abstractmethod
import torch
from torch import autograd


class FixedPointSolver(object):
    """ fixed point solver base class """
    @abstractmethod
    def get_fixed_point(self, init_states, energy_fn):
        """
        :param init_states: A list of tensor
        :param energy_fn: A function that take `states` and return energy for each example
        :return: The fixed point state
        """
        pass


class FixedStepSolver(FixedPointSolver):
    """ Use step size each time """
    def __init__(self, step_size, max_steps=500):
        self.step_size = step_size
        self.max_steps = max_steps

    def get_fixed_point(self, states, energy_fn):
        """ Use fixed step size gradient decsent """
        step = 0
        while step < self.max_steps:
            energy = energy_fn(states)
            #if step % 10 == 0:
            #    print(torch.sum(energy).item())
            grads = autograd.grad(-torch.sum(energy), states)
            for tensor, grad in zip(states, grads):
                tensor[:] = tensor + self.step_size * grad
                tensor[:] = torch.clamp(tensor, 0, 1)
            step += 1
        return states
