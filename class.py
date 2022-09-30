import wandb
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import torchvision
import torchvision.transforms as transforms

train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()])
    

traindata = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=train_transform)
testdata = torchvision.datasets.CIFAR10(root='./', train=False, download=True)

trainloader = DataLoader(traindata)
testloader = DataLoader(testdata)

class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def training_step(self, batch):
        data, gt = batch
        predict = self(data)
        loss = F.cross_entropy(predict, gt)

        self.log('train_loss', loss)
        
        return loss

    def test_step(self, batch):
        data, gt = batch
        predict = self(data)
        loss = F.cross_entropy(predict, gt)

        self.log('test_loss', loss)
        self.log('test_acc', self.accuracy(predict, gt))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


wandb_logger = WandbLogger(name='class',project='class')

net = Net()
trainer = pl.Trainer(logger=wandb_logger, gpus=1)
trainer.fit(model=net, train_dataloaders=trainloader)

trainer.test(dataloaders=testloader)

wandb.finish()