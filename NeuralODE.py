import torch
from torch import nn
from torch import Tensor
from ODEfunc import ODEfunc
from Adjoint import ODEAdjoint


class NeuralODE(nn.Module):
    """
    Wrapper Class to represent the whole Neural ODE
    Need the following component:
    func: the Neural network
    """
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEfunc)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        """
        Use ODE solver to solve the ODE problem,
        The continuous backward propagation is implemented in ODEAdjoint Function.

        :param z0: initial value (bs, z_dim)
        :param t: time step (time, 1, 1)
        :param return_whole_sequence: Bool
        :return: the solved z(t)
        """
        t = t.to(z0)    # Match the data type to z0
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]
