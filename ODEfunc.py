import torch
import torch.nn as nn


class ODEfunc(nn.Module):
    """
    Base Class to represent the Neural Network for the ODE
    i.e. dz/dt = f(z, t; θ)

    To use this Class, subclass it and add User specific Neural Network architecture

    Important function:

    - forward_with_grad
    We need the following things to compute the adjoint state dynamics (da/dt)

    f:      Neural Network output
    dfdz:   Partial derivative of f w.r.t. z at specific time t
    dfdp:   Partial derivative of f w.r.t. parameters at specific time t
    dfdt:   Partial derivative of f w.r.t. t at specific time t

    - flatten_parameters
    Get all the parameter of the Neural Network and flatten it
    """

    def forward_with_grad(self, z, t, grad_outputs):
        """
        Compute the Partial derivative of the Neural Network
        :param z: z value at time t
        :param t: time t
        :param grad_outputs: vector of augmented a [a_z, a_p, a_t]
        :return: Partial derivative (Jacobian) of f = dzdt at direction a
        (f, adfdz, adfdp, adfdt)
        """
        batch_size = z.shape[0]

        # feed forward of f(z,t), by using autograd we can compute the jacobian
        out = self.forward(z, t)
        # direction for autograd
        a = grad_outputs

        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,)
            , (z, t) + tuple(self.parameters())
            , grad_outputs=(a)
            , allow_unused=True
            , retain_graph=True
        )

        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        """
        flatten all parameters inside the Neural Network, so it is easier to compute the gradient with augmented dynamic
        :return: flatten parameters
        """
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

