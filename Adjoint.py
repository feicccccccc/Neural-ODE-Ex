import torch
from NeuralODE import ODEfunc
from ODE_solver import ode_solve
import numpy as np


class ODEAdjoint(torch.autograd.Function):
    """
    Custom made autograd function to perform continuous backpropgation
    define by forward and backward static method
    """
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        """
        forward propagation, solve the given ODE
        dzdt = f(z,t)
        with initial z(t0) = z0
        and find z(t) for t predefined time stamp
        :param ctx: for saving useful information during forward cal
        :param z0: initial z (bs, z_dim)
        :param t: all time stamp (time step, 1, 1)
        :param flat_parameters: flatten parameter
        :param func: the Neural Network represent the ODE function
        :return: (time_len, bs, z_shape)
        """
        assert isinstance(func, ODEfunc)
        bs, *z_shape = z0.size()    # (batch size, z_shape), *for unrolling the tuple
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: (time_len, batch_size, *z_shape)
        Notice the first one is zero if we choose the starting point
        to be the same as the one from true label to start with
        """

        # for enabling Pycharm breakpoint in backward function
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)

        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()

        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is t
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            :param aug_z_i: (bs, n_dim*2 + n_params + 1)
            :param t_i: (bs, 1)
            :return:
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience

        with torch.no_grad():
            # Create placeholders for output gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            # actually I have no idea why we need dLdt and dLdz, but anyway
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                # batch matrix product
                a_t = torch.transpose(dLdz_i.unsqueeze(-1), 1, 2)
                dLdt_i = -torch.bmm(a_t, f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                # Think in terms of chain rule with fix paramter on different node
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] + dLdt_i

                # Pack augmented variable
                # z(t_N), adj_z, 0, adj_t
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = -torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] + dLdt_0
        # forward: (z0, t, parameters, func)
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None