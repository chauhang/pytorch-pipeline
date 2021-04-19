import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
import argparse
import os
from tqdm import tqdm
import requests
from torchtext.utils import download_from_url, extract_archive
import torchtext.datasets as td
import sys
import argparse
import logging
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from pathlib import Path
import subprocess
import json

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-d", "--data_set", required=True, help="first operand")
    ap.add_argument("-o", "--output_path", required=False, help="second operand")
    args = vars(ap.parse_args())

    data_set = json.loads(open(args["data_set"], "r").read())[0]["dataset_url"]

    output_path = args["output_path"]
    Path(output_path).mkdir(parents=True, exist_ok=True)

    dataset_tar = download_from_url(data_set, root="./")
    extracted_files = extract_archive(dataset_tar)

    ag_news_csv = pv.read_csv("ag_news_csv/train.csv")

    pq.write_table(ag_news_csv, os.path.join(output_path, "ag_news_data.parquet"))

    entry_point = ["ls", "-R", output_path]
    run_code = subprocess.run(entry_point, stdout=subprocess.PIPE)
    print(run_code.stdout)

    # for root, dirs, files in os.walk(".", topdown = False):
    #     for name in files:
    #         print(os.path.join(root, name))
    #     for name in dirs:
    #         print(os.path.join(root, name))
