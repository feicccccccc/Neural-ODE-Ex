import numpy as np
import time

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F

from NeuralODE import NeuralODE
from ODEfunc import ODEfunc
from plotter import plot_trajectories

use_cuda = torch.cuda.is_available()
torch.manual_seed(0)


# Helper function, for adding time dimension to the network
def conv3x3kernel(in_feats, out_feats, stride=1):
    """
    Trying to create a time dimension feature?
    :param in_feats:
    :param out_feats:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def add_time(in_tensor, t):
    """
    Add one more dimension to the end of the input tnesor
    :param in_tensor: input
    :param t: time
    :return: concatenated tensor
    """
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


class ConvODEF(ODEfunc):
    """
    Create a ODE func from the input feature
    """
    def __init__(self, dim):
        super(ConvODEF, self).__init__()
        self.conv1 = conv3x3kernel(dim + 1, dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = conv3x3kernel(dim + 1, dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x, t):
        # add time dimension to the data
        # i.e. before: x:(32,64,6,6), t = [0]
        xt = add_time(x, t)
        # after xt:(32,65,6,6)
        xt = self.conv1(xt)
        # after xt: (32,64,6,6)

        h = self.norm1(torch.relu(xt))
        ht = add_time(h, t)
        ht = self.conv2(ht)
        dxdt = self.norm2(torch.relu(ht))
        # same as input vector
        return dxdt


class ContinuousNeuralMNISTClassifier(nn.Module):
    def __init__(self, ode):
        super(ContinuousNeuralMNISTClassifier, self).__init__()
        self.downsampling = nn.Sequential(
            # in channel, out channel, kernel size, stride
            nn.Conv2d(1, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = ode
        self.norm = nn.BatchNorm2d(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # input shape (bs, 1, 28, 28)
        # (batch size, channel, H, W)
        x = self.downsampling(x)
        # output shape (bs, 64, 6, 6)

        # NeuralODE function, time series solver
        # basically add a continuous residual block here
        x = self.feature.forward(x)

        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out


if __name__ == "__main__":
    # Use torchvision to get MNIST data
    import torchvision
    import torchvision.transforms as transforms

    batch_size = 32
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                              download=True)

    # noinspection PyUnresolvedReferences
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # noinspection PyUnresolvedReferences
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model the gradient with a Conv net
    func = ConvODEF(64)
    ode = NeuralODE(func)
    model = ContinuousNeuralMNISTClassifier(ode)
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    # training phase:
    # Enable training mode (for batch norm)
    n_epoch = 5

    for epoch in range(0, n_epoch):

        num_items = 0
        train_losses = []  # keep track of the training loss

        model.train()
        criterion = nn.CrossEntropyLoss()  # notice the labels are not in 1-hot vector form

        # training error
        for batch_idx, (images, labels) in enumerate(train_loader):
            if use_cuda:  # Mode tensor to GPU if available
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_losses += [loss.item()]
            num_items += images.shape[0]
            print("Training on batch: {}".format(batch_idx))

        print('epoch: {}, Train loss: {:.5f}'.format(epoch, np.mean(train_losses)))

        accuracy = 0.0
        num_items = 0

        model.eval()
        criterion = nn.CrossEntropyLoss()
        print(f"Testing...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                output = model(images)
                accuracy += torch.sum(torch.argmax(output, dim=1) == labels).item()
                num_items += images.shape[0]
        accuracy = accuracy * 100 / num_items
        print("Test Accuracy: {:.3f}%".format(accuracy))

    test = 0
