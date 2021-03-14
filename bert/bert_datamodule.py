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

class NewsDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        """
        Performs initialization of tokenizer

        :param reviews: AG news text
        :param targets: labels
        :param tokenizer: bert tokenizer
        :param max_length: maximum length of the news text

        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe

        """
        return len(self.reviews)

    def __getitem__(self, item):
        """
        Returns the review text and the targets of the specified item

        :param item: Index of sample review

        :return: Returns the dictionary of review text, input ids, attention mask, targets
        """
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class BertDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(BertDataModule, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.MAX_LEN = 100
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    @staticmethod
    def process_label(rating):
        rating = int(rating)
        return rating - 1

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """

        num_samples = self.args["num_samples"]

        data_path = self.args["train_glob"]

        print("\n\nTRAIN GLOB")
        print(data_path)
        print("\n\n")

        df_parquet = pq.ParquetDataset(self.args["train_glob"])

        df = df_parquet.read_pandas().to_pandas()

        df.columns = ["label", "title", "description"]
        df.sample(frac=1)
        df = df.iloc[:num_samples]

        df["label"] = df.label.apply(self.process_label)

        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        self.df_train, self.df_test = train_test_split(
            df, test_size=0.1, random_state=RANDOM_SEED, stratify=df["label"]
        )
        self.df_val, self.df_test = train_test_split(
            self.df_test,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=self.df_test["label"],
        )

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        """
        Generic data loader function

        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training

        :return: Returns the constructed dataloader
        """
        ds = NewsDataset(
            reviews=df.description.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_len,
        )

        return DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.test_data_loader
