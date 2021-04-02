import os
import os
import shutil
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
from utils import generate_confusion_matrix, Visualization


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
    bucket_name: str = None,
    folder_name: str = None,
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

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

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

    print("Generating Visualization")
    print("Tensorboard Root Path: {}".format(tensorboard_root))
    Visualization().generate_visualization(tensorboard_root=tensorboard_root)

    if bucket_name:
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


if __name__ == "__main__":

    import sys
    import json

    data_set = json.loads(sys.argv[1])[0]
    output_path = json.loads(sys.argv[2])[0]
    input_parameters = json.loads(sys.argv[3])[0]

    print("INPUT_PARAMETERS:::")
    print(input_parameters)

    tensorboard_root = input_parameters["tensorboard_root"]
    max_epochs = input_parameters["max_epochs"]
    train_batch_size = input_parameters["train_batch_size"]
    val_batch_size = input_parameters["val_batch_size"]
    train_num_workers = input_parameters["train_num_workers"]
    val_num_workers = input_parameters["val_num_workers"]
    learning_rate = input_parameters["learning_rate"]
    accelerator = input_parameters["accelerator"]
    gpus = input_parameters["gpus"]

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
    )
