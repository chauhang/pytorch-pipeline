"""This module is the component of the pipeline for the complete training of the models.
Calls the Executor for the PyTorch Lightning training to start."""
import inspect
import importlib
from typing import Optional, Dict
from pytorch_pipeline.components.trainer.executor import Executor
from pytorch_pipeline.components.base.base_component import BaseComponent
from pytorch_pipeline.types import standard_component_specs


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
        :param data_module_file : The module from which the data module class is inherited.
        :param data_module_args : The arguments of the data module.
        :param module_file_args : The arguments of the model class.
        :param trainer_args : These arguments are specific to the pytorch lightning trainer.
        """

        super(Trainer, self).__init__()
        input_dict = {
            standard_component_specs.TRAINER_MODULE_CLASS: module_file,
            standard_component_specs.TRAINER_DATA_MODULE_CLASS: data_module_file,
        }

        output_dict = {}

        exec_properties = {
            standard_component_specs.TRAINER_DATA_MODULE_ARGS: data_module_args,
            standard_component_specs.TRAINER_MODULE_ARGS: module_file_args,
            standard_component_specs.PTL_TRAINER_ARGS: trainer_args,
        }

        spec = standard_component_specs.TrainerSpec()
        self._validate_spec(
            spec=spec,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
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

            if not model_class:
                raise ValueError(f"Unable to load module_file - {module_file}")

            for cls in inspect.getmembers(
                data_module,
                lambda member: inspect.isclass(member)
                and member.__module__ == data_module.__name__,
            ):
                data_module_class = cls[1]

            if not data_module_class:
                raise ValueError(f"Unable to load data_module_file - {data_module_file}")

            # self.ptl_trainer = Executor().Do(
            #     input_dict=input_dict,
            #     output_dict=output_dict,
            #     exec_properties=exec_properties
            # )
        else:
            raise NotImplementedError(
                "Module file and Datamodule file are mandatory. "
                "Custom training methods are yet to be implemented"
            )
