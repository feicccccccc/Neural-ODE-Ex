import numpy as np

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn  import functional as F

from NeuralODE import NeuralODE
from ODEfunc import ODEfunc
from plotter import plot_trajectories

use_cuda = torch.cuda.is_available()
torch.manual_seed(0)


class LinearODEF(ODEfunc):
    """
    Demo class with linear operation which have a 2x2 weight matrix operation
    """

    def __init__(self, W):
        super(LinearODEF, self).__init__()
        # time independent ODE
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        # x.shape (bs, x_dim)
        # t.shape (bs, 1)
        return self.lin(x)


def generate_trajectory(ode, initial_value, n_points=200, t_max=6.29 * 5):
    """
    Generate the trajectory using the Neural ODE function
    :param ode: Neural ODE
    :param initial_value: z0
    :param n_points: total number of samples
    :param t_max: number of samples
    :return:
    """
    index_np = np.arange(0, n_points, 1, dtype=np.int)
    index_np = np.hstack([index_np[:, None]])  # (n_points, 1)

    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])  # (n_points, 1)

    # (n_points, 1, 1), first dimension is time stamp
    times = torch.from_numpy(times_np[:, :, None]).to(initial_value)

    # (n_points, 1, z_dim) Solved ODE
    observations = ode.forward(initial_value, times, return_whole_sequence=True).detach()

    # add noise to the observation
    observations = observations + torch.randn_like(observations) * 0.01

    return observations, index_np, times_np, times


def create_batch(obs, idx_np, time_np, times, t_max=5):
    """
    TODO: Why only a small batch but not all data?
    Create batches of output from the training ODE
    :param indices: indices of time stamps
    :param times: time stamps
    :param t_max: last time stamp
    :return: observation tensor and corresponding time stamp
    """
    # define length of the random timespan
    min_delta_time = 1.0
    max_delta_time = 4.
    # maximum number of points
    max_points_num = 64

    t0 = np.random.uniform(0, t_max - max_delta_time)
    t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

    idx = sorted(np.random.permutation(idx_np[(time_np > t0) & (time_np < t1)])[:max_points_num])

    obs_ = obs[idx]
    ts_ = times[idx]
    return obs_, ts_


if __name__ == '__main__':
    # set hyper parameter
    n_step = 5000 # gradient descent step
    plot_freq = 50

    # noinspection PyArgumentList
    SpiralFunctionExample = LinearODEF(Tensor([[-0.1, -1.], [1., -0.1]]))
    # RandomLinearODEF = LinearODEF(torch.randn(2, 2))
    # Random seed is not a good way to start, can't converge easily
    RandomLinearODEF = LinearODEF(Tensor([[-1, 1], [-1, 1]]))

    ode_true = NeuralODE(SpiralFunctionExample)
    ode_trained = NeuralODE(RandomLinearODEF)

    # Create data and initial value
    # noinspection PyArgumentList
    z0 = Variable(torch.Tensor([[0.6, 0.3]]))

    # generate the labels from true trajectory
    t_max = 5
    n_points = 11
    # (time index, batch size, dimension of output)
    obs, index_np, time_np, times = generate_trajectory(ode_true, z0, n_points=n_points, t_max=t_max)

    # Train Neural ODE
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.01)

    # Training cycle:
    for i in range(n_step):
        # Forward Pass for training Model. Get trajectory of random timespan
        obs_batch, ts_batch = create_batch(obs, index_np, time_np, times, t_max=t_max)
        z_batch = ode_trained(obs_batch[0], ts_batch, return_whole_sequence=True)

        # compare label with model output
        loss = F.mse_loss(z_batch, obs_batch.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Plot the trajectory
        # plot_trajectories(obs=[obs], times=[time_np], trajs=[obs_]

        if i % plot_freq == 0:
            print("step: {}, loss: {}".format(i, loss))
            z_p = ode_trained(obs[0], times, return_whole_sequence=True)
            plot_trajectories(obs=[obs], times=[times], trajs=[z_p])

    test = 0
