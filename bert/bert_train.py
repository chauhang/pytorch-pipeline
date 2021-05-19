# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.metrics import Accuracy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchtext.datasets as td
from torchtext.utils import download_from_url, extract_archive
from transformers import AdamW, BertModel, BertTokenizer
from pytorch_lightning.loggers import TensorBoardLogger
import pyarrow.parquet as pq
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import shutil


class BertNewsClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(BertNewsClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.2)
        # assigning labels
        self.class_names = ["World", "Sports", "Business", "Sci/Tech"]
        n_classes = len(self.class_names)

        self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 512)
        self.out = nn.Linear(512, n_classes)
        # self.bert_model.embedding = self.bert_model.embeddings
        # self.embedding = self.bert_model.embeddings

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def compute_bert_outputs(
        self, model_bert, embedding_input, attention_mask=None, head_mask=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones(embedding_input.shape[0], embedding_input.shape[1]).to(
                embedding_input
            )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(model_bert.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(model_bert.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * model_bert.config.num_hidden_layers

        encoder_outputs = model_bert.encoder(
            embedding_input, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = model_bert.pooler(sequence_output)
        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        return outputs

    def forward(self, input_ids, attention_mask=None):
        """
        :param input_ids: Input data
        :param attention_maks: Attention mask value

        :return: output - Type of news for the given news snippet
        """
        embedding_input = self.bert_model.embeddings(input_ids)
        # output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.compute_bert_outputs(self.bert_model, embedding_input)
        pooled_output = outputs[1]
        output = F.relu(self.fc1(pooled_output))
        output = self.drop(output)
        output = self.out(output)
        return output

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch["attention_mask"].to(self.device)
        targets = train_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, targets)
        self.train_acc(y_hat, targets)
        self.log("train_acc", self.train_acc.compute())
        self.log("train_loss", loss)
        return {"loss": loss, "acc": self.train_acc.compute()}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        input_ids = test_batch["input_ids"].to(self.device)
        attention_mask = test_batch["attention_mask"].to(self.device)
        targets = test_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), targets.cpu())
        self.test_acc(y_hat, targets)
        self.log("test_acc", self.test_acc.compute())
        return {"test_acc": torch.tensor(test_acc)}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """

        input_ids = val_batch["input_ids"].to(self.device)
        attention_mask = val_batch["attention_mask"].to(self.device)
        targets = val_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, targets)
        self.val_acc(y_hat, targets)
        self.log("val_acc", self.val_acc.compute())
        self.log("val_loss", loss, sync_dist=True)
        return {"val_step_loss": loss, "acc": self.val_acc.compute()}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = AdamW(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]


def train_model(
    train_glob: str,
    tensorboard_root: str,
    max_epochs: int,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    learning_rate: int,
    accelerator: str,
    model_save_path: str,
):
    """
    method to train and validate the model

    :param train_glob: Input sentences from the batch
    :param tensorboard_root: Path to save the tensorboard logs
    :param max_epochs: Maximum number of epochs
    :param num_samples: Maximum number of samples to train the model
    :param batch_size: Number of samples ImportError: sys.meta_path is None, Python is likely shutting downper batch
    :param num_workers: Number of cores to train the model
    :param learning_rate: Learning rate used to train the model
    :param accelerator: single or multi GPU
    :param model_save_path: Path for the model to be saved
    :param bucket_name: Name of the S3 bucket
    :param folder_name: Name of the folder to write in S3
    :param webapp_path: Path to save the web content
    """

    if accelerator == "None":
        accelerator = None

    dict_args = {
        "train_glob": train_glob,
        "max_epochs": max_epochs,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": learning_rate,
        "accelerator": accelerator,
    }

    from bert_datamodule import BertDataModule

    dm = BertDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = BertNewsClassifier(**dict_args)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    if os.path.exists(os.path.join(tensorboard_root, "bert_lightning_kubeflow")):
        shutil.rmtree(os.path.join(tensorboard_root, "bert_lightning_kubeflow"))

    Path(tensorboard_root).mkdir(parents=True, exist_ok=True)

    # Tensorboard root name of the logging directory
    tboard = TensorBoardLogger(tensorboard_root, "bert_lightning_kubeflow")

    Path(model_save_path).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="bert_news_classification_{epoch:02d}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        logger=tboard,
        accelerator=accelerator,
        callbacks=[lr_logger, early_stopping, checkpoint_callback],
        checkpoint_callback=True,
        max_epochs=max_epochs,
    )

    trainer.fit(model, dm)
    trainer.test()
    torch.save(model.state_dict(), os.path.join(model_save_path, "bert.pth"))


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
    num_samples = input_parameters["num_samples"]
    batch_size = input_parameters["batch_size"]
    num_workers = input_parameters["num_workers"]
    learning_rate = input_parameters["learning_rate"]
    accelerator = input_parameters["accelerator"]

    train_model(
        data_set,
        tensorboard_root,
        max_epochs,
        num_samples,
        batch_size,
        num_workers,
        learning_rate,
        accelerator,
        output_path,
    )
