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
            standard_component_specs.TRAINER_MODULE_FILE: module_file,
            standard_component_specs.TRAINER_DATA_MODULE_FILE: data_module_file,
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

            Executor().Do(
                input_dict=input_dict, output_dict=output_dict, exec_properties=exec_properties
            )

            self.ptl_trainer = output_dict.get(standard_component_specs.PTL_TRAINER_OBJ, "None")
            self.output_dict = output_dict
        else:
            raise NotImplementedError(
                "Module file and Datamodule file are mandatory. "
                "Custom training methods are yet to be implemented"
            )
