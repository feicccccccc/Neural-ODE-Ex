import math


def ode_solve(z0, t0, t1, f):
    """
    Simple Euler ODE initial Value solver
    :param z0: initial time
    :param t0: initial time stamp
    :param t1: target time stamp
    :param f: ODE function
    :return: z1
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0) / h_max).max().item())

    h = (t1 - t0) / n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z
