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


MAR_GENERATION_CONFIG = "mar_config"
MAR_GENERATION_SAVE_PATH = "mar_save_path"
CONFIG_PROPERTIES_SAVE_PATH = "config_prop_save_path"

VIZ_MLPIPELINE_UI_METADATA = "mlpipeline_ui_metadata"
VIZ_MLPIPELINE_METRICS = "mlpipeline_metrics"
VIZ_CONFUSION_MATRIX_DICT = "confusion_matrix_dict"
VIZ_TEST_ACCURACY = "test_accuracy"

VIZ_MARKDOWN = "markdown"
VIZ_MARKDOWN_DICT_SOURCE = "source"
VIZ_MARKDOWN_DICT_STORAGE = "storage"

VIZ_CONFUSION_MATRIX_ACTUALS = "actuals"
VIZ_CONFUSION_MATRIX_PREDS = "preds"
VIZ_CONFUSION_MATRIX_CLASSES = "classes"
VIZ_CONFUSION_MATRIX_URL = "url"

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


class MarGenerationSpec:  # pylint: disable=R0903
    """Mar Specification class.
    For validating the parameter 'type' .
    """

    INPUT_DICT = {
        MAR_GENERATION_CONFIG: Parameters(type=dict),
    }

    OUTPUT_DICT = {}

    EXECUTION_PROPERTIES = {
        MAR_GENERATION_SAVE_PATH: Parameters(type=str, optional=True),
    }

class VisualizationSpec:
    """Visualization Specification class.
    For validating the parameter 'type'
    """
    INPUT_DICT = {
        VIZ_CONFUSION_MATRIX_DICT: Parameters(type=dict, optional=True),
        VIZ_TEST_ACCURACY: Parameters(type=float, optional=True),
        VIZ_MARKDOWN: Parameters(type=dict, optional=True),
    }

    OUTPUT_DICT = {}

    EXECUTION_PROPERTIES = {
        VIZ_MLPIPELINE_UI_METADATA: Parameters(type=str, optional=True),
        VIZ_MLPIPELINE_METRICS: Parameters(type=str, optional=True),
    }

    MARKDOWN_DICT = {
        VIZ_MARKDOWN_DICT_STORAGE: Parameters(type=str, optional=False),
        VIZ_MARKDOWN_DICT_SOURCE: Parameters(type=dict, optional=False),
    }

    CONFUSION_MATRIX_DICT = {
        VIZ_CONFUSION_MATRIX_ACTUALS: Parameters(type=list, optional=False),
        VIZ_CONFUSION_MATRIX_PREDS: Parameters(type=list, optional=False),
        VIZ_CONFUSION_MATRIX_CLASSES: Parameters(type=list, optional=False),
        VIZ_CONFUSION_MATRIX_URL: Parameters(type=str, optional=False),
    }
