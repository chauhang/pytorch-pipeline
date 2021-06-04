import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from pytorch_kfp_components.components.trainer.component import Trainer
from pytorch_kfp_components.components.mar.component import MarGeneration

# Argument parser for user defined paths
parser = ArgumentParser()

parser.add_argument(
    "--tensorboard_root",
    type=str,
    default="output/tensorboard",
    help="Tensorboard Root path (default: output/tensorboard)",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="output",
    help="Path to save model checkpoints (default: output/train/models)",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="iris.pt",
    help="Name of the model to be saved as (default: iris.pt)",
)

parser = pl.Trainer.add_argparse_args(parent_parser=parser)

args = vars(parser.parse_args())


if not args["max_epochs"]:
    max_epochs = 5
else:
    max_epochs = args["max_epochs"]


args["max_epochs"] = max_epochs

trainer_args = {}

# Initiating the training process
trainer = Trainer(
    module_file="iris_classification.py",
    data_module_file="iris_data_module.py",
    module_file_args=args,
    data_module_args=None,
    trainer_args=trainer_args,
)

# Mar file generation

mar_config = {
    "MODEL_NAME": "iris_classification",
    "MODEL_FILE": "tests/iris/iris_classification.py",
    "HANDLER": "tests/iris/iris_handler.py",
    "SERIALIZED_FILE": os.path.join(args["checkpoint_dir"], args["model_name"]),
    "VERSION": "1",
    "EXPORT_PATH": args["checkpoint_dir"],
    "CONFIG_PROPERTIES": "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/config.properties",
}


MarGeneration(mar_config=mar_config, mar_save_path=args["checkpoint_dir"])
