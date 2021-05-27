# pylint: disable=R0913
# pylint: disable=R0903
# pylint: disable=E1003

"""This module is the component of the pipeline for the complete training of the models.
Calls the Executor for the PyTorch Lighthing training to start."""

import inspect
import importlib
from typing import Optional, Dict
from pytorch_pipeline.components.base.base_component import BaseComponent
from pytorch_pipeline.components.trainer.executor import Executor


class Trainer(BaseComponent):
    """Initializes the Trainer class."""

    def __init__(
        self,
        module_file: Optional = None,
        data_module_file: Optional = None,
        data_module_args: Optional[Dict] = None,
        module_file_args: Optional[Dict] = None,
        trainer_args: Optional[Dict] = None,
    ):
        """
        Initializes the PyTorch Lightning training process.

        :param module_file : The module to inherit the model class for training.
        :param data_module_file : The module from which the data module clas is inherited.
        :param data_module_args : The arguments of the data module.
        :param module_file_args : The arguments of the model class.
        :param trainer_args : These arguments are specific to the trainer.
        """
        super(BaseComponent, self).__init__()
        if [bool(module_file)] != 1:
            raise ValueError(
                "Exactly one of 'module_file', 'trainer_fn', or 'run_fn' must be " "supplied."
            )

        if module_file and data_module_file:
            # Both module file and data module file are present

            model_class = None
            data_module_class = None

            class_module = importlib.import_module(module_file.split(".")[0])
            data_module = importlib.import_module(data_module_file.split(".")[0])

            for cls in inspect.getmembers(
                class_module,
                lambda member: inspect.isclass(member)
                and member.__module__ == class_module.__name__,
            ):
                model_class = cls[1]

            for cls in inspect.getmembers(
                data_module,
                lambda member: inspect.isclass(member)
                and member.__module__ == data_module.__name__,
            ):
                data_module_class = cls[1]

            print(model_class, data_module_class)

            self.ptl_trainer = Executor().Do(
                model_class=model_class,
                data_module_class=data_module_class,
                data_module_args=data_module_args,
                module_file_args=module_file_args,
                trainer_args=trainer_args,
            )
