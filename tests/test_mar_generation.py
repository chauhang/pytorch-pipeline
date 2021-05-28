import os
import re
import subprocess
import tempfile

import pytest

from pytorch_pipeline.components.mar.mar_generation import MarGeneration

IRIS_DIR = "tests/iris"
EXPORT_PATH = tempfile.mkdtemp()
print(f"Export path: {EXPORT_PATH}")

MAR_CONFIG = {
    "MODEL_NAME": "iris_classification",
    "MODEL_FILE": f"{IRIS_DIR}/iris_classification.py",
    "HANDLER": f"{IRIS_DIR}/iris_handler.py",
    "SERIALIZED_FILE": f"{EXPORT_PATH}/iris.pt",
    "VERSION": "1",
    "EXPORT_PATH": EXPORT_PATH,
    "CONFIG_PROPERTIES": "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/config.properties",
}

MANDATORY_ARGS = [
    "MODEL_NAME",
    "SERIALIZED_FILE",
    "MODEL_FILE",
    "HANDLER",
    "VERSION",
    "CONFIG_PROPERTIES",
]

OPTIONAL_ARGS = ["EXTRA_FILES", "REQUIREMENTS_FILE"]

DEFAULT_HANDLERS = [
    "image_classifier",
    "text_classifier",
    "image_segmenter",
    "object_detector",
]


def generate_mar_file(config, save_path):
    MarGeneration(mar_config=config).generate_mar_file(save_path)
    mar_path = os.path.join(EXPORT_PATH, "iris_classification.mar")
    config_properties = os.path.join(EXPORT_PATH, "config.properties")
    assert os.path.exists(mar_path)
    assert os.path.exists(config_properties)

    os.remove(mar_path)
    os.remove(config_properties)


def test_mar_generation_empty_config():
    empty_mar_config = {}
    tmp_dir = tempfile.mkdtemp()

    exception_msg = re.escape(
        f"Mar config cannot be empty. Mandatory arguments are {MANDATORY_ARGS}"
    )
    with pytest.raises(Exception, match=exception_msg):
        MarGeneration(mar_config=empty_mar_config).generate_mar_file(tmp_dir)


@pytest.mark.parametrize("mandatory_key", MANDATORY_ARGS)
def test_mar_generation_mandatory_params_missing(mandatory_key):
    tmp_value = MAR_CONFIG[mandatory_key]
    MAR_CONFIG[mandatory_key] = ""
    MAR_CONFIG.pop(mandatory_key)

    tmp_dir = tempfile.mkdtemp()
    excpetion_msg = re.escape(
        f"Following Mandatory keys are missing in the config file ['{mandatory_key}']"
    )
    with pytest.raises(Exception, match=excpetion_msg):
        MarGeneration(mar_config=MAR_CONFIG).generate_mar_file(tmp_dir)

    MAR_CONFIG[mandatory_key] = tmp_value


def test_mar_generation_success():
    cmd = [
        "python",
        "iris_pytorch.py",
        "--checkpoint_dir",
        EXPORT_PATH,
    ]

    cwd = os.getcwd()
    os.chdir(IRIS_DIR)
    subprocess.run(cmd)
    os.chdir(cwd)
    generate_mar_file(config=MAR_CONFIG, save_path=EXPORT_PATH)


@pytest.mark.parametrize("handler", DEFAULT_HANDLERS)
def test_mar_generation_default_handlers(handler):
    tmp_value = MAR_CONFIG["HANDLER"]
    MAR_CONFIG["HANDLER"] = handler
    generate_mar_file(config=MAR_CONFIG, save_path=EXPORT_PATH)
    MAR_CONFIG["HANDLER"] = tmp_value


@pytest.mark.parametrize("optional_arg", OPTIONAL_ARGS)
def test_mar_generation_optional_arguments(optional_arg):
    new_file, filename = tempfile.mkstemp()

    MAR_CONFIG[optional_arg] = os.path.join(os.getcwd(), filename)

    generate_mar_file(config=MAR_CONFIG, save_path=EXPORT_PATH)

    MAR_CONFIG.pop(optional_arg)