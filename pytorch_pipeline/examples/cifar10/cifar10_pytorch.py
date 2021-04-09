import pytorch_lightning as pl
from pytorch_pipeline.components.trainer.component import Trainer
from argparse import ArgumentParser


parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parent_parser=parser)
args = vars(parser.parse_args())

max_epochs = args.get("max_epochs", 1)

module_file_args = {

"max_epochs" : max_epochs

}

data_module_args = {
    "train_glob": "output/processing"
}

print("Arguments: ", args)

trainer = Trainer(
    module_file="cifar10_train.py",
    data_module_file="cifar10_datamodule.py",
    module_file_args=module_file_args,
    data_module_args=data_module_args
)

