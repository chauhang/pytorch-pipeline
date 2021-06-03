import pytest
import os
import json
from pytorch_pipeline.components.visualization.component import Visualization
from unittest.mock import patch
import tempfile

metdata_dir = tempfile.mkdtemp()


@pytest.fixture(scope="class")
def viz_params():
    MARKDOWN_PARAMS = {
        "storage": "dummy-storage",
        "source": {"dummy_key": "dummy_value"},
    }

    VIZ_PARAMS = {
        "mlpipeline_ui_metadata": os.path.join(
            metdata_dir, "mlpipeline_ui_metadata.json"
        ),
        "mlpipeline_metrics": os.path.join(metdata_dir, "mlpipeline_metrics"),
        "confusion_matrix_dict": {},
        "test_accuracy": 99.05,
        "markdown": MARKDOWN_PARAMS,
    }
    return VIZ_PARAMS


def generate_visualization(viz_params: dict):
    viz_obj = Visualization(
        mlpipeline_ui_metadata=viz_params["mlpipeline_ui_metadata"],
        mlpipeline_metrics=viz_params["mlpipeline_metrics"],
        confusion_matrix_dict=viz_params["confusion_matrix_dict"],
        test_accuracy=viz_params["test_accuracy"],
        markdown=viz_params["markdown"],
    )

    return viz_obj.output_dict


@pytest.mark.parametrize(
    "viz_key",
    [
        "confusion_matrix_dict",
        "test_accuracy",
        "markdown",
    ],
)
def test_invalid_type_viz_params(viz_params, viz_key):
    viz_params[viz_key] = "dummy"
    if viz_key == "test_accuracy":
        expected_type = "<class 'float'>"
    else:
        expected_type = "<class 'dict'>"
    expected_exception_msg = f"{viz_key} must be of type {expected_type} but received as {type(viz_params[viz_key])}"
    with pytest.raises(TypeError, match=expected_exception_msg):
        generate_visualization(viz_params)


@pytest.mark.parametrize(
    "viz_key",
    [
        "mlpipeline_ui_metadata",
        "mlpipeline_metrics",
    ],
)
def test_invalid_type_metadata_path(viz_params, viz_key):

    viz_params[viz_key] = ["dummy"]
    expected_exception_msg = f"{viz_key} must be of type <class 'str'> but received as {type(viz_params[viz_key])}"
    with pytest.raises(TypeError, match=expected_exception_msg):
        generate_visualization(viz_params)


@pytest.mark.parametrize(
    "viz_key",
    [
        "mlpipeline_ui_metadata",
        "mlpipeline_metrics",
    ],
)
def test_default_metadata_path(viz_params, viz_key):
    viz_params[viz_key] = None
    expected_output = {
        "mlpipeline_ui_metadata": "/mlpipeline-ui-metadata.json",
        "mlpipeline_metrics": "/mlpipeline-metrics.json",
    }
    with patch(
        "test_visualization.generate_visualization",
        return_value=expected_output,
    ):
        output_dict = generate_visualization(viz_params)
    assert output_dict == expected_output


def test_custom_metadata_path(viz_params, tmpdir):
    metadata_ui_path = os.path.join(str(tmpdir), "mlpipeline_ui_metadata.json")
    metadata_metrics_path = os.path.join(str(tmpdir), "mlpipeline_metrics.json")
    viz_params["mlpipeline_ui_metadata"] = metadata_ui_path
    viz_params["mlpipeline_metrics"] = metadata_metrics_path
    output_dict = generate_visualization(viz_params)
    assert output_dict is not None
    assert output_dict["mlpipeline_ui_metadata"] == metadata_ui_path
    assert output_dict["mlpipeline_metrics"] == metadata_metrics_path
    assert os.path.exists(metadata_ui_path)
    assert os.path.exists(metadata_metrics_path)


def test_setting_all_keys_to_none(viz_params):
    for key in viz_params.keys():
        viz_params[key] = None

    expected_exception_msg = r"Any one of these keys should be set - confusion_matrix_dict, test_accuracy, markdown"
    with pytest.raises(ValueError, match=expected_exception_msg):
        generate_visualization(viz_params)


def test_accuracy_metric(viz_params):
    output_dict = generate_visualization(viz_params)
    assert output_dict is not None
    metadata_metric_file = viz_params["mlpipeline_metrics"]
    assert os.path.exists(metadata_metric_file)
    with open(metadata_metric_file) as fp:
        data = json.load(fp)
    assert data["metrics"][0]["numberValue"] == viz_params["test_accuracy"]


def test_markdown_storage_invalid_datatype(viz_params):
    viz_params["markdown"]["storage"] = ["test"]
    expected_exception_msg = (
        r"storage must be of type <class 'str'> but received as {}".format(
            type(viz_params["markdown"]["storage"])
        )
    )
    with pytest.raises(TypeError, match=expected_exception_msg):
        generate_visualization(viz_params)


def test_markdown_source_invalid_datatype(viz_params):
    viz_params["markdown"]["source"] = "test"
    expected_exception_msg = (
        r"source must be of type <class 'dict'> but received as {}".format(
            type(viz_params["markdown"]["source"])
        )
    )
    with pytest.raises(TypeError, match=expected_exception_msg):
        generate_visualization(viz_params)


@pytest.mark.parametrize(
    "markdown_key",
    [
        "source",
        "storage",
    ],
)
def test_markdown_source_missing_key(viz_params, markdown_key):
    del viz_params["markdown"][markdown_key]
    expected_exception_msg = r"Missing mandatory key - {}".format(markdown_key)
    with pytest.raises(ValueError, match=expected_exception_msg):
        generate_visualization(viz_params)


def test_markdown_success(viz_params):
    output_dict = generate_visualization(viz_params)
    assert output_dict is not None
    assert "mlpipeline_ui_metadata" in output_dict
    assert os.path.exists(output_dict["mlpipeline_ui_metadata"])
    with open(output_dict["mlpipeline_ui_metadata"]) as fp:
        data = fp.read()
    assert "dummy_key" in data
    assert "dummy_value" in data


def test_different_storage_value(viz_params):
    viz_params["markdown"]["storage"] = "inline"
    output_dict = generate_visualization(viz_params)
    assert output_dict is not None
    assert "mlpipeline_ui_metadata" in output_dict
    assert os.path.exists(output_dict["mlpipeline_ui_metadata"])
    with open(output_dict["mlpipeline_ui_metadata"]) as fp:
        data = fp.read()
    assert "inline" in data


def test_multiple_metadata_appends(viz_params):
    if os.path.exists(viz_params["mlpipeline_ui_metadata"]):
        os.remove(viz_params["mlpipeline_ui_metadata"])

    if os.path.exists(viz_params["mlpipeline_metrics"]):
        os.remove(viz_params["mlpipeline_metrics"])
    generate_visualization(viz_params)
    generate_visualization(viz_params)
    output_dict = generate_visualization(viz_params)
    assert output_dict is not None
    assert "mlpipeline_ui_metadata" in output_dict
    assert os.path.exists(output_dict["mlpipeline_ui_metadata"])
    with open(output_dict["mlpipeline_ui_metadata"]) as fp:
        data = json.load(fp)
    assert len(data["outputs"]) == 3
