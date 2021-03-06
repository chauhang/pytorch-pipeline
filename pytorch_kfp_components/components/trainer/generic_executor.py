#!/usr/bin/env/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generic Executor Class."""
import importlib
import inspect
from pytorch_kfp_components.components.base.base_executor import BaseExecutor
from pytorch_kfp_components.types import standard_component_specs


class GenericExecutor(BaseExecutor):
    """Generic Executor Class that does nothing."""

    def Do(self, input_dict: dict, output_dict: dict, exec_properties: dict):
        # TODO: Code to train pretrained model
        pass

    def _GetFnArgs(
        self, input_dict: dict, output_dict: dict, execution_properties: dict
    ):
        """Gets the input/output/execution properties from the dictionary.

        Args:
            input_dict : The dictionary of inputs.Example :
                        model file, data module file
            output_dict :
            exec_properties : A dict of execution properties including
                            data_module_args,trainer_args, module_file_args
        Returns:
            module_file : The model file name
            data_module_file : A data module file name
            trainer_args: A dictionary of trainer args
            module_file_args : A dictionary of model specific args
            data_module_args : A dictionary of data module args.
        """
        module_file = input_dict.get(
            standard_component_specs.TRAINER_MODULE_FILE
        )
        data_module_file = input_dict.get(
            standard_component_specs.TRAINER_DATA_MODULE_FILE
        )
        trainer_args = execution_properties.get(
            standard_component_specs.PTL_TRAINER_ARGS
        )
        module_file_args = execution_properties.get(
            standard_component_specs.TRAINER_MODULE_ARGS
        )
        data_module_args = execution_properties.get(
            (standard_component_specs.TRAINER_DATA_MODULE_ARGS)
        )
        return (
            module_file,
            data_module_file,
            trainer_args,
            module_file_args,
            data_module_args,
        )

    def derive_model_and_data_module_class(
        self, module_file: str, data_module_file: str
    ):
        """Derives the model file and data modul file.

        Args :
            module_file : A model file name (type:str)
            data_module_file : A data module file name (type:str)

        Returns :
            model_class : The model class
            data_module_class : The data module class.

        Raises :
            ValueError: If the model file or data module file is empty.
        """
        model_class = None
        data_module_class = None

        class_module = importlib.import_module(module_file.split(".")[0])
        data_module = importlib.import_module(data_module_file.split(".")[0])

        for cls in inspect.getmembers(
                class_module,
                lambda member: inspect.isclass(member) and member.__module__ ==
                class_module.__name__,
        ):
            model_class = cls[1]

        if not model_class:
            raise ValueError(f"Unable to load module_file - {module_file}")

        for cls in inspect.getmembers(
                data_module,
                lambda member: inspect.isclass(member) and member.__module__ ==
                data_module.__name__,
        ):
            data_module_class = cls[1]

        if not data_module_class:
            raise ValueError(
                f"Unable to load data_module_file - {data_module_file}"
            )

        return model_class, data_module_class
