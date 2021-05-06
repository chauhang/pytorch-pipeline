import pytorch_lightning as pl
import os
from pytorch_pipeline.components.trainer.component import Trainer
from pytorch_pipeline.components.mar.mar_generation import MarGeneration
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


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
    default="output/train/models",
    help="Path to save model checkpoints (default: output/train/models)",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default="output/processing",
    help="Cifar10 Dataset path (default: output/processing)",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="bert.pth",
    help="Name of the model to be saved as (default: bert.pth)",
)


parser = pl.Trainer.add_argparse_args(parent_parser=parser)

args = vars(parser.parse_args())


# Enabling Tensorboard Logger, ModelCheckpoint, Earlystopping

lr_logger = LearningRateMonitor()
tboard = TensorBoardLogger(args["tensorboard_root"])
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=True)
checkpoint_callback = ModelCheckpoint(
    dirpath=args["checkpoint_dir"],
    filename="cifar10_{epoch:02d}",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

if not args["max_epochs"]:
    max_epochs = 1
else:
    max_epochs = args["max_epochs"]


# Setting the trainer specific arguments
trainer_args = {
    "logger": tboard,
    "checkpoint_callback": True,
    "max_epochs": max_epochs,
    "callbacks": [lr_logger, early_stopping, checkpoint_callback],
}


# if "profiler" in args:
#     trainer_args["profiler"] = args["profiler"]

# Setting the datamodule specific arguments
data_module_args = {
    "train_glob": args["dataset_path"],
}


# Initiating the training process
trainer = Trainer(
    module_file="bert_train.py",
    data_module_file="bert_datamodule.py",
    module_file_args=parser,
    data_module_args=data_module_args,
    trainer_args=trainer_args,
)


# Mar file generation

mar_config = {
    "MODEL_NAME": "bert_test",
    "MODEL_FILE": "pytorch_pipeline/examples/bert/bert_train.py",
    "HANDLER": "pytorch_pipeline/examples/bert/bert_handler.py",
    "SERIALIZED_FILE": os.path.join(args["checkpoint_dir"], args["model_name"]),
    "VERSION": "1",
    "EXPORT_PATH": args["checkpoint_dir"],
    "CONFIG_PROPERTIES": "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/bert/config.properties",
    "EXTRA_FILES": "pytorch_pipeline/examples/bert/bert-base-uncased-vocab.txt,pytorch_pipeline/examples/bert/index_to_name.json,pytorch_pipeline/examples/bert/wrapper.py",
}


MarGeneration(mar_config=mar_config).generate_mar_file(mar_save_path=args["checkpoint_dir"])
