import os

import torch
from torchmetrics import MetricCollection, Accuracy

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split, DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from CapsuleNet import CapsuleNet, margin_loss


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
        return DataLoader(self.mnist_train, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)


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
    model = BasicVisionSystem(lr=0.01)
    trainer = Trainer(max_epochs=10, gpus='7,', precision=16, callbacks=RichProgressBar())
    trainer.fit(model, data)