#!/usr/bin/env/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for trainer component."""
import os
import shutil
import sys
import tempfile
from copy import deepcopy

import pytest
import pytorch_lightning

from pytorch_pipeline.components.trainer.component import Trainer

dirname, filename = os.path.split(os.path.abspath(__file__))
IRIS_DIR = os.path.join(dirname, "iris")
sys.path.insert(0, IRIS_DIR)

MODULE_FILE_ARGS = {"lr": 0.1}
TRAINER_ARGS = {"max_epochs": 5}
DATA_MODULE_ARGS = {"num_workers": 2}
trainer_params = {
    "module_file": "iris_classification.py",
    "data_module_file": "iris_data_module.py",
    "module_file_args": MODULE_FILE_ARGS,
    "data_module_args": DATA_MODULE_ARGS,
    "trainer_args": TRAINER_ARGS,
}

MANDATORY_ARGS = [
    "module_file",
    "data_module_file",
]
OPTIONAL_ARGS = ["module_file_args", "data_module_args", "trainer_args"]

DEFAULT_MODEL_NAME = "model_state_dict.pth"
DEFAULT_SAVE_PATH = f"/tmp/{DEFAULT_MODEL_NAME}"


def invoke_training(trainer_params):  # pylint: disable=W0621
    """This function invokes the training process."""
    trainer = Trainer(
        module_file=trainer_params["module_file"],
        data_module_file=trainer_params["data_module_file"],
        module_file_args=trainer_params["module_file_args"],
        trainer_args=trainer_params["trainer_args"],
        data_module_args=trainer_params["data_module_args"],
    )
    return trainer


@pytest.mark.parametrize("mandatory_key", MANDATORY_ARGS)
def test_mandatory_keys_type_check(mandatory_key):
    """Tests the uncexpected 'type' of mandatory args.

    Args:
        mandatory_key : mandatory arguments for inivoking training
    """
    trainer_dict = deepcopy(trainer_params)
    test_input = ["input_path"]
    trainer_dict[mandatory_key] = test_input
    expected_exception_msg = f"{mandatory_key} must be of type <class 'str'> but received as {type(test_input)}"
    with pytest.raises(TypeError, match=expected_exception_msg):
        invoke_training(trainer_params=trainer_dict)


@pytest.mark.parametrize("optional_key", OPTIONAL_ARGS)
def test_optional_keys_type_check(optional_key):
    """Tests the unexpected 'type' of optional args.

    Args:
        optional_key: optional arguments for invoking training
    """
    trainer_dict = deepcopy(trainer_params)
    test_input = "test_input"
    trainer_dict[optional_key] = test_input
    expected_exception_msg = f"{optional_key} must be of type <class 'dict'> but received as {type(test_input)}"
    with pytest.raises(TypeError, match=expected_exception_msg):
        invoke_training(trainer_params=trainer_dict)


@pytest.mark.parametrize("input_key", MANDATORY_ARGS + ["module_file_args"])
def test_mandatory_params(input_key):
    """Test for empty mandatory arguments.

    Args:
        input_key: name of the mandatory arg for training
    """
    trainer_dict = deepcopy(trainer_params)
    trainer_dict[input_key] = None
    expected_exception_msg = (
        f"{input_key} is not optional. Received value: {trainer_dict[input_key]}"
    )
    with pytest.raises(ValueError, match=expected_exception_msg):
        invoke_training(trainer_params=trainer_dict)


def test_data_module_args_optional():
    """Test for empty optional argument : data module args"""
    trainer_dict = deepcopy(trainer_params)
    trainer_dict["data_module_args"] = None
    invoke_training(trainer_params=trainer_dict)
    assert os.path.exists(DEFAULT_SAVE_PATH)
    os.remove(DEFAULT_SAVE_PATH)


def test_trainer_args_none():
    """Test for empty trainer specific arguments."""
    trainer_dict = deepcopy(trainer_params)
    trainer_dict["trainer_args"] = None
    expected_exception_msg = r"trainer_args must be a dict"
    with pytest.raises(TypeError, match=expected_exception_msg):
        invoke_training(trainer_params=trainer_dict)


def test_training_success():
    """Test the training success case with all required args."""
    trainer = invoke_training(trainer_params=trainer_params)
    assert os.path.exists(DEFAULT_SAVE_PATH)
    os.remove(DEFAULT_SAVE_PATH)
    assert hasattr(trainer, "ptl_trainer")
    assert isinstance(
        trainer.ptl_trainer, pytorch_lightning.trainer.trainer.Trainer
    )

def test_training_success_with_custom_model_name():
    """Test for successful training with custom model name."""
    tmp_dir = tempfile.mkdtemp()
    custom_model_name_dict = deepcopy(trainer_params)
    custom_model_name_dict["module_file_args"]["checkpoint_dir"] = tmp_dir
    custom_model_name_dict["module_file_args"]["model_name"] = "iris.pth"
    invoke_training(trainer_params=custom_model_name_dict)
    assert "iris.pth" in os.listdir(tmp_dir)
    shutil.rmtree(tmp_dir)


def test_training_success_with_empty_module_file_args():
    """Test for successful training with empty module file args."""
    empty_args_dict = deepcopy(trainer_params)
    empty_args_dict["module_file_args"] = {}
    invoke_training(trainer_params=trainer_params)


def test_training_success_with_empty_trainer_args():
    """Test for successful training with empty trainer args."""
    tmp_dir = tempfile.mkdtemp()
    empty_args_dict = deepcopy(trainer_params)
    empty_args_dict["module_file_args"]["max_epochs"] = 5
    empty_args_dict["module_file_args"]["checkpoint_dir"] = tmp_dir
    empty_args_dict["trainer_args"] = {}
    invoke_training(trainer_params=empty_args_dict)
    assert DEFAULT_MODEL_NAME in os.listdir(tmp_dir)
    shutil.rmtree(tmp_dir)


def test_training_success_with_empty_data_module_args():
    """Test for successful training with empty data module args."""
    tmp_dir = tempfile.mkdtemp()
    tmp_trainer_parms = deepcopy(trainer_params)
    tmp_trainer_parms["module_file_args"]["checkpoint_dir"] = tmp_dir
    tmp_trainer_parms["data_module_args"] = None
    invoke_training(trainer_params=tmp_trainer_parms)

    assert DEFAULT_MODEL_NAME in os.listdir(tmp_dir)
    shutil.rmtree(tmp_dir)


def test_trainer_output():
    """Test for successful training with proper saving of training output."""
    tmp_dir = tempfile.mkdtemp()
    tmp_trainer_parms = deepcopy(trainer_params)
    tmp_trainer_parms["module_file_args"]["checkpoint_dir"] = tmp_dir
    trainer = invoke_training(trainer_params=tmp_trainer_parms)

    assert hasattr(trainer, "output_dict")
    assert trainer.output_dict is not None
    assert trainer.output_dict["model_save_path"] == os.path.join(
        tmp_dir, DEFAULT_MODEL_NAME
    )
    assert isinstance(
        trainer.output_dict["ptl_trainer"],
        pytorch_lightning.trainer.trainer.Trainer,
    )
