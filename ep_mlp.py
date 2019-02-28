""" An mlp that use EP to train.
https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop/blob/master/model.py
"""
import torch
from torch.nn import init
from torch import matmul
from torch import autograd


def flatten(vars):
    """
    :param vars: A list of tensor. Each is [bsz, shape]
    :return: flattend_var: [bsz, shape1 + shape2 + ...]
             shapes: A list of torch.Size.
    """
    shapes = [var.shape[1:] for var in vars]
    flatten_var = [var.view(-1, var.shape[1:].numel()) for var in vars]
    return torch.cat(flatten_var, 1), shapes


def unflatten(flattened, shapes):
    """ Reverse of flatten """
    vars = torch.split(flattened, [shape.numel() for shape in shapes], dim=1)
    return [var.view(var.size(0), *shape) for var, shape in zip(vars, shapes)]


class Linear:
    """ A linear layer in EP """
    def __init__(self, in_features, out_features, bias=True, device=None):
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.weight = torch.Tensor(out_features, in_features).to(device=self.device)

        if bias:
            self.bias_in = torch.Tensor(in_features).to(device=device)
            self.bias_out = torch.Tensor(out_features).to(device=device)
        else:
            self.bias_in = None
            self.bias_out = None

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=1)
        self.weight.requires_grad = True
        if self.bias_out is not None:
            self.bias_in.zero_()
            self.bias_out.zero_()
            self.bias_in.requires_grad = True
            self.bias_out.requires_grad = True

    def get_energy(self, inputs, outputs):
        """
        :param inputs: [b, inp_size]
        :param outputs: [b, out_size]
        :return: [b, 1]
        """
        # Question: Do I need a 0.5 here? From the paper there is a 0.5, but in the repo there isn't
        neg_energy = matmul(matmul(outputs[:, None, :], self.weight), inputs[:, :, None]).squeeze(-1)
        if self.bias_in is not None:
            neg_energy += matmul(inputs, self.bias_in[:, None]) + matmul(outputs, self.bias_out[:, None])
        return -neg_energy

    def parameters(self):
        params = [self.weight]
        if self.bias_in is not None:
            params += [self.bias_in, self.bias_out]
        return params

    def set_gradients(self, free_inp, free_out, clamp_inp, clamp_out):
        """ Given the free/clamp phase result. Perform backward """
        params = self.parameters()

        # Free phase grad
        free_energy = self.get_energy(free_inp, free_out)
        free_grads = autograd.grad(torch.mean(free_energy), params)

        # Clamp phase grad
        clamp_energy = self.get_energy(clamp_inp, clamp_out)
        clamp_grads = autograd.grad(torch.mean(clamp_energy), params)
        for param, free_grad, clamp_grad in zip(params, free_grads, clamp_grads):
            param.grad = clamp_grad - free_grad


class EPMLP(object):

    def __init__(self, in_size, out_size, hidden_sizes, non_linear=None, device=None):
        """
        :param in_size: int
        :param out_size: int
        :param hidden_sizes: list of int or empty
        :param non_linear: non linear function. E.g., torch.nn.functional.sigmoid
        """
        self._in_size = in_size
        self._out_size = out_size
        self._hidden_sizes = hidden_sizes

        # Initialize weights
        layer_sizes = [in_size] + hidden_sizes + [out_size]
        self._layers = []
        for idx in range(len(layer_sizes) - 1):
            self._layers += [Linear(in_features=layer_sizes[idx],
                                    out_features=layer_sizes[idx + 1],
                                    device=device)]
        self._non_linear = non_linear if non_linear is not None \
            else lambda x: torch.clamp(x, min=0, max=1)

    def get_state_shapes(self):
        """ The shape of state """
        return [layer.out_features for layer in self._layers]

    def get_energy(self, inp, states):
        """
        :param inp: [bsz, inp_size]
        :param out: [bsz, out_size]
        :return: energy [bsz, 1]
        """
        # Non linear first
        acts = [inp] + states
        acts = [self._non_linear(act) for act in acts]

        # Energy for each layer
        energy = 0
        for idx, layer in enumerate(self._layers):
            energy += layer.get_energy(acts[idx], acts[idx + 1])

        # The regularize therm
        for act in acts:
            energy += 0.5 * torch.sum(act.pow(2), -1, keepdim=True)
        return energy

    def get_cost(self, states, label):
        """ l2 loss """
        out = states[-1]
        out = self._non_linear(out)
        return (out - label).pow(2).sum(-1, keepdim=True)

    def init_out(self, batch_size, requires_grad=False):
        """ Return a random output """
        return torch.rand([batch_size, self._out_size], requires_grad=requires_grad).to(device=self.device)

    def init_hiddens(self, batch_size, requires_grad=False):
        """ Return hidden units """
        return [torch.rand([batch_size, self._layers[idx].out_features],
                           requires_grad=requires_grad).to(device=self.device)
                for idx in range(len(self._layers) - 1)]

    def get_init_states(self, batch_size, hidden_units=None, out=None, requires_grad=False):
        """ Return a list of tensors """
        hidden_units = [t.clone().to(device=t.device) for t in hidden_units] if hidden_units is not None \
            else self.init_hiddens(batch_size, requires_grad)
        out = out.clone().to(device=out.device) if out is not None \
            else self.init_out(batch_size, requires_grad)

        # Put to corresponding device
        for t in hidden_units:
            t = t.to(device=self.device)
        out.to(device=self.device)
        return hidden_units + [out]

    @property
    def device(self):
        return self._layers[0].device

    def free_phase(self, inp, solver, out=None, hidden_units=None):
        """ Perform free_phase
        :return free_phase states
        """
        bsz = inp.size(0)
        init_states = self.get_init_states(bsz, hidden_units, out, requires_grad=True)
        fp_states = solver.get_fixed_point(init_states,
                                           lambda states: self.get_energy(inp, states))
        return fp_states

    def clamp_phase(self, inp, label, solver, beta, out=None, hidden_units=None):
        """ Perform weakly clamped """
        bsz = inp.size(0)
        init_states = self.get_init_states(bsz, hidden_units, out, requires_grad=True)
        fp_states = solver.get_fixed_point(init_states,
                                           lambda states: self.get_energy(inp, states) +
                                                          beta * self.get_cost(states, label))
        return fp_states

    def set_gradients(self, inp, free_states, clamp_states):
        """ Set the gradient given free states and clamp states """
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                free_in = inp
                clamp_in = inp
            else:
                free_in = free_states[idx - 1]
                clamp_in = clamp_states[idx - 1]
            free_out = free_states[idx]
            clamp_out = clamp_states[idx]
            layer.set_gradients(free_in, free_out, clamp_in, clamp_out)

    def parameters(self):
        """ List of parameters """
        res = []
        for layer in self._layers:
            res += layer.parameters()
        return res
