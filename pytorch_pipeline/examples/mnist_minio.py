import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from minio import Minio
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class LitMNIST(pl.LightningModule):

    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


model = LitMNIST()

tensorboard_root = os.getcwd()

if os.path.exists(os.path.join(tensorboard_root, "mnist_lightning_kubeflow")):
    shutil.rmtree(os.path.join(tensorboard_root, "mnist_lightning_kubeflow"))

Path(tensorboard_root).mkdir(parents=True, exist_ok=True)
# Tensorboard root name of the logging directory
tboard = TensorBoardLogger(tensorboard_root, "mnist_lightning_kubeflow")

trainer = pl.Trainer(max_epochs=1, logger=tboard)
trainer.fit(model)
trainer.test()


###########################
# USING MINIO #
###########################


client = Minio(
    '172.17.0.2:9000', access_key="minioadmin", secret_key="minioadmin", secure=False
)

bucket_name = "kubeflow-dataset"
folder_name = "mnist"

for path in Path("mnist_lightning_kubeflow/version_0/").rglob("*"):
    if not path.is_dir():
        client.fput_object(
            bucket_name=bucket_name,
            object_name=os.path.join(
                folder_name, os.path.relpath(start='mnist_lightning_kubeflow/version_0', path=path)
            ),
            file_path=path,
        )


