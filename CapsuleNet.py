import torch
from torch import nn
from torch.nn import functional as F


def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor


def criterion(x, y_hat, target, reconstructions):
    return margin_loss(y_hat, target) + reconstruction_loss(x, reconstructions)


def margin_loss(x, labels, size_average=True):
    batch_size = x.size(0)

    v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

    left = F.relu(0.9 - v_c).view(batch_size, -1)
    right = F.relu(v_c - 0.1).view(batch_size, -1)

    loss = labels * left + 0.5 * (1.0 - labels) * right
    loss = loss.sum(dim=1).mean()

    return loss


def reconstruction_loss(x, reconstructions):
    batch_size = reconstructions.size(0)
    loss = F.mse_loss(reconstructions.view(batch_size, -1), x.view(batch_size, -1))
    return loss * 0.0005


class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=32, out_channels=8, kernel_size=(9, 9), stride=2):
        super(PrimaryCapsule, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels * out_channels, kernel_size, stride)
        self.reshape = lambda x: x.view(x.size(0), -1, out_channels)

    def forward(self, x):
        out = squash(self.reshape(self.conv(x)))
        return out


class DigitCapsule(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCapsule, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1).to(x.device)

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 256, 9, 1, 0), nn.ReLU())
        self.primary_capsule = PrimaryCapsule()
        self.digit_capsule = DigitCapsule()

    def forward(self, x):
        x = self.conv1(x)
        x = self.primary_capsule(x)
        x = self.digit_capsule(x)
        return x