import os

import torch
from torchmetrics import MetricCollection, Accuracy

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split, DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning import LightningDataModule, LightningModule, Trainer


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data", img_size=(28, 28), batch_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.train_transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, pin_memory=True, persistent_workers=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)


def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor


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


class BasicVisionSystem(LightningModule):
    def __init__(self, lr=0.001):
        super(BasicVisionSystem, self).__init__()

        self.model = CapsuleNet()
        self.criterion = margin_loss
        self.lr = lr

        metric = MetricCollection({'top@1': Accuracy(top_k=1), 'top@5': Accuracy(top_k=5)})
        self.train_metric = metric.clone(prefix='train_')
        self.valid_metric = metric.clone(prefix='valid_')

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.shared_step(*batch, self.train_metric)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(*batch, self.valid_metric)

    def shared_step(self, x, y, metric):
        y_hat = self.model(x)
        loss = self.criterion(y_hat, torch.eye(10).to(y.device)[y])
        self.log_dict(metric((y_hat**2).sum(dim=(-1, -2)), y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.lr)
        scheduler = OneCycleLR(optimizer, max_lr=self.lr, total_steps=10 * int(55000 / 512 + 1))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


if __name__ == '__main__':
    data = MNISTDataModule(img_size=(28, 28), batch_size=512)
    model = BasicVisionSystem(lr=0.001)
    trainer = Trainer(max_epochs=10, gpus='0,', precision=16, callbacks=RichProgressBar())
    trainer.fit(model, data)