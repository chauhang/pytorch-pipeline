import pytest
import os
import json
from pytorch_pipeline.components.visualization.component import Visualization
from pytorch_pipeline.types import standard_component_specs
from unittest.mock import patch
import tempfile

metdata_dir = tempfile.mkdtemp()


@pytest.fixture(scope="class")
def viz_params():
    VIZ_PARAMS = {
        standard_component_specs.VIZ_MLPIPELINE_UI_METADATA: os.path.join(
            metdata_dir, "mlpipeline_ui_metadata.json"
        ),
        standard_component_specs.VIZ_MLPIPELINE_METRICS: os.path.join(
            metdata_dir, "mlpipeline_metrics"
        ),
        standard_component_specs.VIZ_CONFUSION_MATRIX_DICT: {},
        standard_component_specs.VIZ_TEST_ACCURACY: 99.05,
        standard_component_specs.VIZ_MARKDOWN: {"source": "dummy_value", "storage": "dummy"},
    }
    return VIZ_PARAMS


def generate_visualization(viz_params: dict):
    viz_obj = Visualization(
        mlpipeline_ui_metadata=viz_params[standard_component_specs.VIZ_MLPIPELINE_UI_METADATA],
        mlpipeline_metrics=viz_params[standard_component_specs.VIZ_MLPIPELINE_METRICS],
        confusion_matrix_dict=viz_params[standard_component_specs.VIZ_CONFUSION_MATRIX_DICT],
        test_accuracy=viz_params[standard_component_specs.VIZ_TEST_ACCURACY],
        markdown=viz_params[standard_component_specs.VIZ_MARKDOWN],
    )

    return viz_obj.output_dict


@pytest.mark.parametrize(
    "viz_key",
    [
        standard_component_specs.VIZ_CONFUSION_MATRIX_DICT,
        standard_component_specs.VIZ_TEST_ACCURACY,
        standard_component_specs.VIZ_MARKDOWN,
    ],
)
def test_invalid_type_viz_params(viz_params, viz_key):
    viz_params[viz_key] = "dummy"
    if viz_key == standard_component_specs.VIZ_TEST_ACCURACY:
        expected_type = "<class 'float'>"
    else:
        expected_type = "<class 'dict'>"
    expected_exception_msg = (
        f"{viz_key} must be of type {expected_type} but received as {type(viz_params[viz_key])}"
    )
    with pytest.raises(TypeError, match=expected_exception_msg):
        generate_visualization(viz_params)


@pytest.mark.parametrize(
    "viz_key",
    [
        standard_component_specs.VIZ_MLPIPELINE_UI_METADATA,
        standard_component_specs.VIZ_MLPIPELINE_METRICS,
    ],
)
def test_invalid_type_metadata_path(viz_params, viz_key):

    viz_params[viz_key] = ["dummy"]
    expected_exception_msg = (
        f"{viz_key} must be of type <class 'str'> but received as {type(viz_params[viz_key])}"
    )
    with pytest.raises(TypeError, match=expected_exception_msg):
        generate_visualization(viz_params)


@pytest.mark.parametrize(
    "viz_key",
    [
        standard_component_specs.VIZ_MLPIPELINE_UI_METADATA,
        standard_component_specs.VIZ_MLPIPELINE_METRICS,
    ],
)
def test_default_metadata_path(viz_params, viz_key):
    viz_params[viz_key] = None
    expected_output = {
        standard_component_specs.VIZ_MLPIPELINE_UI_METADATA: "/mlpipeline-ui-metadata.json",
        standard_component_specs.VIZ_MLPIPELINE_METRICS: "/mlpipeline-metrics.json",
    }
    with patch("test_visualization.generate_visualization", return_value=expected_output):
        output_dict = generate_visualization(viz_params)
    assert output_dict == expected_output


def test_custom_metadata_path(viz_params, tmpdir):
    metadata_ui_path = os.path.join(str(tmpdir), "mlpipeline_ui_metadata.json")
    metadata_metrics_path = os.path.join(str(tmpdir), "mlpipeline_metrics.json")
    viz_params[standard_component_specs.VIZ_MLPIPELINE_UI_METADATA] = metadata_ui_path
    viz_params[standard_component_specs.VIZ_MLPIPELINE_METRICS] = metadata_metrics_path
    output_dict = generate_visualization(viz_params)
    assert output_dict is not None
    assert output_dict[standard_component_specs.VIZ_MLPIPELINE_UI_METADATA] == metadata_ui_path
    assert output_dict[standard_component_specs.VIZ_MLPIPELINE_METRICS] == metadata_metrics_path
    assert os.path.exists(metadata_ui_path)
    assert os.path.exists(metadata_metrics_path)


def test_setting_all_keys_to_none(viz_params):
    for key in viz_params.keys():
        viz_params[key] = None

    expected_exception_msg = (
        r"Any one of these keys should be set - confusion_matrix_dict, test_accuracy, markdown"
    )
    with pytest.raises(ValueError, match=expected_exception_msg):
        generate_visualization(viz_params)


def test_accuracy_metric(viz_params):
    output_dict = generate_visualization(viz_params)
    assert output_dict is not None
    metadata_metric_file = viz_params[standard_component_specs.VIZ_MLPIPELINE_METRICS]
    assert os.path.exists(metadata_metric_file)
    with open(metadata_metric_file) as fp:
        data = json.load(fp)
    assert (
        data["metrics"][0]["numberValue"] == viz_params[standard_component_specs.VIZ_TEST_ACCURACY]
    )
