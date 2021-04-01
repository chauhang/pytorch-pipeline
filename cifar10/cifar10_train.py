import os
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torchvision import models
from utils import generate_confusion_matrix


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(CIFAR10Classifier, self).__init__()
        self.model_conv = models.resnet50(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features
        num_classes = 10
        self.model_conv.fc = nn.Linear(num_ftrs, num_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.preds = []
        self.target = []

    def forward(self, x):
        out = self.model_conv(x)
        return out

    def training_step(self, train_batch, batch_idx):
        if batch_idx == 0:
            self.reference_image = (train_batch[0][0]).unsqueeze(0)
            # self.reference_image.resize((1,1,28,28))
            print("\n\nREFERENCE IMAGE!!!")
            print(self.reference_image.shape)
        x, y = train_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y)
        self.log("train_loss", loss)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc.compute())
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y)
        if self.args["accelerator"] is not None:
            self.log("test_loss", loss, sync_dist=True)
        else:
            self.log("test_loss", loss)
        self.test_acc(y_hat, y)
        self.preds += y_hat.tolist()
        self.target += y.tolist()

        self.log("test_acc", self.test_acc.compute())
        return {"test_acc": self.test_acc.compute()}

    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y)
        if self.args["accelerator"] is not None:
            self.log("val_loss", loss, sync_dist=True)
        else:
            self.log("val_loss", loss)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc.compute())
        return {"val_step_loss": loss, "val_loss": loss}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]

    def makegrid(self, output, numrows):
        outer = torch.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if j == numrows:
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c

    def showActivations(self, x):

        # logging reference image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x[0][0]), self.current_epoch, dataformats="HW"
        )

        # logging layer 1 activations
        out = self.model_conv.conv1(x)
        c = self.makegrid(out, 4)
        self.logger.experiment.add_image("layer 1", c, self.current_epoch, dataformats="HW")

    def training_epoch_end(self, outputs):
        self.showActivations(self.reference_image)

        # Logging graph
        if self.current_epoch == 0:
            sampleImg = torch.rand((1, 3, 64, 64))
            self.logger.experiment.add_graph(CIFAR10Classifier(), sampleImg)


def train_model(
    train_glob: str,
    gpus: int,
    tensorboard_root: str,
    max_epochs: int,
    train_batch_size: int,
    val_batch_size: int,
    train_num_workers: int,
    val_num_workers: int,
    learning_rate: int,
    accelerator: str,
    model_save_path: str,
    bucket_name: str,
    folder_name: str,
):

    if accelerator == "None":
        accelerator = None
    if train_batch_size == "None":
        train_batch_size = None
    if val_batch_size == "None":
        val_batch_size = None

    dict_args = {
        "train_glob": train_glob,
        "max_epochs": max_epochs,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "train_num_workers": train_num_workers,
        "val_num_workers": val_num_workers,
        "lr": learning_rate,
        "accelerator": accelerator,
    }

    from cifar10_datamodule import CIFAR10DataModule

    dm = CIFAR10DataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = CIFAR10Classifier(**dict_args)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)

    Path(model_save_path).mkdir(parents=True, exist_ok=True)

    if len(os.listdir(model_save_path)) > 0:
        for filename in os.listdir(model_save_path):
            filepath = os.path.join(model_save_path, filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="cifar10_{epoch:02d}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    if os.path.exists(os.path.join(tensorboard_root, "cifar10_lightning_kubeflow")):
        shutil.rmtree(os.path.join(tensorboard_root, "cifar10_lightning_kubeflow"))

    Path(tensorboard_root).mkdir(parents=True, exist_ok=True)

    # Tensorboard root name of the logging directory
    tboard = TensorBoardLogger(tensorboard_root, "cifar10_lightning_kubeflow")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        gpus=gpus,
        logger=tboard,
        checkpoint_callback=True,
        max_epochs=max_epochs,
        callbacks=[lr_logger, early_stopping, checkpoint_callback],
        accelerator=accelerator,
    )

    trainer.fit(model, dm)
    trainer.test()
    torch.save(model.state_dict(), os.path.join(model_save_path, "resnet.pth"))

    generate_confusion_matrix(
        actuals=trainer.model.target,
        preds=trainer.model.preds,
        output_path=os.path.join(model_save_path, "confusion_matrix.csv"),
    )

    s3 = boto3.resource("s3")
    bucket_name = bucket_name
    folder_name = folder_name
    bucket = s3.Bucket(bucket_name)
    s3_path = "s3://" + bucket_name + "/" + folder_name

    for obj in bucket.objects.filter(Prefix=folder_name + "/"):
        s3.Object(bucket.name, obj.key).delete()

    for event_file in os.listdir(tensorboard_root + "/cifar10_lightning_kubeflow/version_0"):
        s3.Bucket(bucket_name).upload_file(
            tensorboard_root + "/cifar10_lightning_kubeflow/version_0/" + event_file,
            folder_name + "/" + event_file,
            ExtraArgs={"ACL": "public-read"},
        )

    with open("logdir.txt", "w") as f:
        f.write(s3_path)


def add_parser_arguments(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="/pvc/output/processing",
        help="Dataset path (default: /pvc/output/processing)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/pvc/output/train/models",
        help="Path to write the output (default: /pvc/output/train/models)",
    )

    parser.add_argument(
        "--tensorboard_root",
        type=str,
        default="/pvc/output/train/tensorboard",
        help="Path to log tensorboard artifacts (default: /pvc/output/train/tensorboard)",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        help="Number of epochs to run (default: 5)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of gpus (default: 0)",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Train Batch Size (default: None)",
    )

    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Validation batch size Batch Size (default: None)",
    )

    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=4,
        help="Number of workers for training (default: 4)",
    )

    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=4,
        help="Number of workers for validation (default: 4)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Accelerator (default: None)",
    )

    parser.add_argument(
        "--bucket_name",
        type=str,
        default="kubeflow-dataset",
        help="S3 bucket name (default: kubeflow-dataset)",
    )

    parser.add_argument(
        "--s3_folder_path",
        type=str,
        default="Cifar10Viz",
        help="S3 folder path (default: Cifar10Viz)",
    )

    return parser


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)

    parser = add_parser_arguments(parser)

    args = vars(parser.parse_args())

    data_set = args["dataset"]
    output_path = args["output_path"]

    tensorboard_root = args["tensorboard_root"]
    max_epochs = args["max_epochs"]
    train_batch_size = args["train_batch_size"]
    val_batch_size = args["val_batch_size"]
    train_num_workers = args["train_num_workers"]
    val_num_workers = args["val_num_workers"]
    learning_rate = args["learning_rate"]
    accelerator = args["accelerator"]
    gpus = args["gpus"]
    bucket_name = args["bucket_name"]
    folder_name = args["s3_folder_path"]

    train_model(
        train_glob=data_set,
        model_save_path=output_path,
        tensorboard_root=tensorboard_root,
        max_epochs=max_epochs,
        gpus=gpus,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        train_num_workers=train_num_workers,
        val_num_workers=val_num_workers,
        learning_rate=learning_rate,
        accelerator=accelerator,
        bucket_name=bucket_name,
        folder_name=folder_name,
    )
