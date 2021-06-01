#!/usr/bin/env/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Module for defining standard specifications and validation of parameter
type."""

TRAINER_MODULE_FILE = "module_file"
TRAINER_DATA_MODULE_FILE = "data_module_file"
TRAINER_DATA_MODULE_ARGS = "data_module_args"
TRAINER_MODULE_ARGS = "module_file_args"
PTL_TRAINER_ARGS = "trainer_args"

TRAINER_MODEL_SAVE_PATH = "model_save_path"
PTL_TRAINER_OBJ = "ptl_trainer"


class Parameters:  # pylint: disable=R0903
    """Parameter class to match the desired type."""
    def __init__(self, type=None, optional=False):  # pylint: disable=redefined-builtin
        self.type = type
        self.optional = optional


class TrainerSpec:  # pylint: disable=R0903
    """Trainer Specification class.

    For validating the parameter 'type' .
    """
    INPUT_DICT = {
        TRAINER_MODULE_FILE: Parameters(type=str),
        TRAINER_DATA_MODULE_FILE: Parameters(type=str),
    }

    OUTPUT_DICT = {}

    EXECUTION_PROPERTIES = {
        TRAINER_DATA_MODULE_ARGS: Parameters(type=dict, optional=True),
        TRAINER_MODULE_ARGS: Parameters(type=dict),
        PTL_TRAINER_ARGS: Parameters(type=dict, optional=True),
    }