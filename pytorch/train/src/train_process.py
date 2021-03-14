import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import requests
import torch
import torch.nn.functional as F
from common.pytorch_component import ComponentMetadata, PytorchComponent
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchtext.utils import download_from_url, extract_archive
from torchvision import transforms
from tqdm import tqdm
from transformers import AdamW, BertModel, BertTokenizer

from train_process_spec import (PytorchTrainInputs, PytorchTrainOutputs,
                                PytorchTrainSpec)


@ComponentMetadata(
    name="Pytorch - Training Job",
    description="Perform training in pytorch",
    spec=PytorchTrainSpec,
)
class PytorchTrainComponent(PytorchComponent):
    """Pytorch component for training."""

    def Do(self, spec: PytorchTrainSpec):
        super().Do(spec.inputs, spec.outputs, spec.output_paths)

    def _run_pipeline_step(
        self, inputs: PytorchTrainInputs, outputs: PytorchTrainOutputs,
    ):        
        print("Inside run pipeline!!!! for training step")

        if inputs.source_code:
            print("Inside source code block!!!")
            print(inputs.source_code[0])
            for i in inputs.source_code:
                download_from_url(i, root=inputs.source_code_path[0])
            print("download successfull")

            entry_point=["ls", "-R", "/pvc/input"]
            run_code = subprocess.run(entry_point, stdout=subprocess.PIPE)
            print("Checking downloaded file!!!")
            print(run_code.stdout)

        if inputs.container_entrypoint:
            print("Inside entry point container block")
            entry_point = inputs.container_entrypoint
            entry_point.append(json.dumps(inputs.input_data))
            entry_point.append(json.dumps(inputs.output_data))
            entry_point.append(json.dumps(inputs.input_parameters))
            run_code = subprocess.run(entry_point, stdout=subprocess.PIPE)
            print(run_code.stdout)

            
if __name__ == "__main__":

    import sys

    spec = PytorchTrainSpec(sys.argv[1:])
    component = PytorchTrainComponent()
    component.Do(spec)

    
    