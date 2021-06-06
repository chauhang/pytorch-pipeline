#
# from unittest.mock import patch
# from unittest import mock
# from pytorch_kfp_components.components.minio.component import MinIO
# from pytorch_kfp_components.components.minio.executor import Executor
# from minio import Minio
#
# def test_minio_upload():
#     # client = Minio("localhost:9000")
#     # client.fput_object(bucket_name="test", object_name="image.png", file_path="/home/ubuntu/Documents/facebook/phase2/kubeflow/pytorch-pipeline/image.png")
#     with mock.patch.object(Executor, "_initiate_minio_client") as client:
#         # client.side_effect = ValueError
#         # mock.patch.object(Executor, "_use_default_config")
#         # mock.patch("Minio.client")
#         # client = mock_client.return_value
#         # client.fput_object(bucket_name="test", object_name="image.png", file_path="/home/ubuntu/Documents/facebook/phase2/kubeflow/pytorch-pipeline/image.png")
#         print("******************** \n\n\n " , dir(client))
#         MinIO(bucket_name="test", destination="test1", source="/home/ubuntu/Documents/facebook/phase2/kubeflow/pytorch-pipeline/image.png", endpoint="localhost:9000")
#         print(client.assert_called_once())
#         print(client.mock_calls)

import mock
import pytest
import tempfile
import urllib3
import os
from pytorch_kfp_components.components.minio.component import MinIO
from pytorch_kfp_components.components.minio.executor import Executor

tmpdir = tempfile.mkdtemp()

with open(os.path.join(str(tmpdir), "dummy.txt"), "w") as fp:
    fp.write("dummy")

@pytest.fixture(scope="class")
def minio_inputs():
    minio_inputs = {
        "bucket_name": "dummy",
        "source": f"{tmpdir}/dummy.txt",
        "destination": "dummy.txt",
        "endpoint": "localhost:9000"

    }
    return minio_inputs


def upload_to_minio(minio_inputs):
    MinIO(source=minio_inputs["source"], bucket_name=minio_inputs["bucket_name"], destination=minio_inputs["destination"], endpoint=minio_inputs["endpoint"])


@pytest.mark.parametrize(
    "key",
    ["source", "bucket_name", "destination", "endpoint"],
)
def test_minio_variables_invalid_type(minio_inputs, key):
    minio_inputs[key] = ["test"]
    expected_exception_msg = f"{key} must be of type <class 'str'> but received as {type(minio_inputs[key])}"
    with pytest.raises(TypeError, match=expected_exception_msg):
        upload_to_minio(minio_inputs)

@pytest.mark.parametrize(
    "key",
    ["source", "bucket_name", "destination", "endpoint"],
)
def test_minio_mandatory_param(minio_inputs, key):
    minio_inputs[key] = None
    expected_exception_msg = f"{key} is not optional. Received value: {minio_inputs[key]}"
    with pytest.raises(ValueError, match=expected_exception_msg):
        upload_to_minio(minio_inputs)


def test_missing_access_key(minio_inputs):
    os.environ["MINIO_SECRET_KEY"] = "dummy"
    expected_exception_msg = "Environment variable MINIO_ACCESS_KEY not found"
    with pytest.raises(ValueError, match=expected_exception_msg):
        upload_to_minio(minio_inputs)

    os.environ.pop("MINIO_SECRET_KEY")


def test_missing_secret_key(minio_inputs):
    os.environ["MINIO_ACCESS_KEY"] = "dummy"
    expected_exception_msg = "Environment variable MINIO_SECRET_KEY not found"
    with pytest.raises(ValueError, match=expected_exception_msg):
        upload_to_minio(minio_inputs)
    os.environ.pop("MINIO_ACCESS_KEY")


def test_unreachable_endpoint(minio_inputs):
    os.environ["MINIO_ACCESS_KEY"] = "dummy"
    os.environ["MINIO_SECRET_KEY"] = "dummy"
    with pytest.raises(Exception, match="Max retries exceeded with url*"):
        upload_to_minio(minio_inputs)


def test_invalid_file_path(minio_inputs):
    minio_inputs["source"] = "dummy"
    expected_exception_msg = f"Input path - {minio_inputs['source']} does not exists"
    with pytest.raises(ValueError, match=expected_exception_msg):
        upload_to_minio(minio_inputs)


def test_minio_upload_file(minio_inputs):
    with mock.patch.object(Executor, "upload_artifacts_to_minio") as client:
        client.return_value = []
        upload_to_minio(minio_inputs)
        client.asser_called_once()


def test_minio_upload_folder(minio_inputs):
    minio_inputs["source"] = tmpdir
    with mock.patch.object(Executor, "upload_artifacts_to_minio") as client:
        client.return_value = []
        upload_to_minio(minio_inputs)
        client.asser_called_once()

