import pytorch_lightning as pl
import os
from argparse import ArgumentParser
from pytorch_pipeline.components.ax.ax_hpo import AxOptimization
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_pipeline.components.mar.mar_generation import MarGeneration
from pytorch_pipeline.components.visualization.component import Visualization


# Argument parser for user defined paths
parser = ArgumentParser()

parser.add_argument(
    "--tensorboard_root",
    type=str,
    default="output/tensorboard",
    help="Tensorboard Root path (default: output/tensorboard)",
)


parser.add_argument(
    "--model_name",
    type=str,
    default="mnist.pth",
    help="Name of the model to be saved as (default: mnist.pth)",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="output",
    help="Name of the model to be saved as (default: resnet.pth)",
)

parser.add_argument(
    "--summary_url",
    type=str,
    help="Minio url to generate Ax-Experiment-Summary)",
)

parser.add_argument(
    "--mlpipeline_ui_metadata",
    type=str,
    help="Path to write mlpipeline-ui-metadata.json",
)

parser = pl.Trainer.add_argparse_args(parent_parser=parser)

args = vars(parser.parse_args())
print("/n/n This is args[max_epochs]",args["max_epochs"])


# Enabling Tensorboard Logger, ModelCheckpoint, Earlystopping

Path(args["tensorboard_root"]).mkdir(parents=True, exist_ok=True)
tboard = TensorBoardLogger(args["tensorboard_root"])

if args["max_epochs"]==None:
    args["max_epochs"] = 1
max_epochs = args["max_epochs"]
print("\n\n")
print("This is max_epochs",args["max_epochs"])
# Setting the trainer specific arguments
trainer_args = {
    "logger": tboard,
    "max_epochs": max_epochs,
}


model_file_name = "mnist_train.py"
data_module_file_name = "mnist_datamodule.py"

# params_file = os.path.join("/pytorch_pipeline/examples/mnist/parameters.json")
# with open(params_file) as f:
#    data = f.read()

ax_params = [
    {"name": "lr", "type": "range", "bounds": [1e-3, 0.15], "log_scale": True},
    {"name": "weight_decay", "type": "range", "bounds": [1e-4, 1e-3]},
    {"name": "momentum", "type": "range", "bounds": [0.7, 1.0]},
]

total_trials = 2
ax_hpo = AxOptimization(total_trials, ax_params)

ax_hpo.run_ax_get_best_parameters(
    module_file_args=args,
    data_module_args=None,
    trainer_args=trainer_args,
    model_file_name=model_file_name,
    data_module_file_name=data_module_file_name,
)
best_model_name = "mnist_best.pth"
print("This", best_model_name)

mar_config = {
    "MODEL_NAME": "mnist_best",
    "MODEL_FILE": "pytorch_pipeline/examples/mnist/mnist_train.py",
    "HANDLER": "pytorch_pipeline/examples/mnist/mnist_handler.py",
    "SERIALIZED_FILE": os.path.join(args["checkpoint_dir"], best_model_name),
    "VERSION": "1",
    "EXPORT_PATH": args["checkpoint_dir"],
    "CONFIG_PROPERTIES": "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/mnist_ax/config.properties",
}


MarGeneration(mar_config=mar_config).generate_mar_file(mar_save_path=args["checkpoint_dir"])

columns = ax_hpo.columns


table_dict = {
    'type': 'table',
    'storage': 'minio',
    'format': 'csv',
    'header': [x for x in columns],
    'source': args["summary_url"]
      }

visualization = Visualization(
    mlpipeline_ui_metadata=args["mlpipeline_ui_metadata"],
    table=table_dict,
)