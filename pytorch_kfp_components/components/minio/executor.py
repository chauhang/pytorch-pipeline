import os
from pytorch_kfp_components.components.base.base_executor import BaseExecutor
from pytorch_kfp_components.types import standard_component_specs
from minio import Minio
import urllib3


class Executor(BaseExecutor):
    def __init__(self):
        super(Executor, self).__init__()

    def _initiate_minio_client(self, minio_config: dict):
        minio_host = minio_config["HOST"]
        access_key = minio_config["ACCESS_KEY"]
        secret_key = minio_config["SECRET_KEY"]
        client = Minio(
            minio_host,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        return client

    def _read_minio_creds(self, endpoint: str):
        if "MINIO_ACCESS_KEY" not in os.environ:
            raise ValueError("Environment variable MINIO_ACCESS_KEY not found")

        if "SECRET_KEY" not in os.environ:
            raise ValueError("Environment variable SECRET_KEY not found")

        minio_config = {
            "HOST": endpoint,
            "ACCESS_KEY": os.environ["MINIO_ACCESS_KEY"],
            "SECRET_KEY": os.environ["SECRET_KEY"],
        }

        return minio_config

    def upload_artifacts_to_minio(
        self,
        client: Minio,
        source: str,
        destination: str,
        bucket_name: str,
        output_dict: dict,
    ):
        print(f"source {source} destination {destination}")
        try:
            result = client.fput_object(
                bucket_name=bucket_name,
                file_path=source,
                object_name=destination,
            )
            if result[0] is None:
                raise RuntimeError(
                    "Upload failed - source: {}  destination - {} bucket_name - {}".format(
                        source, destination, bucket_name
                    )
                )
            else:
                output_dict[destination] = {
                    "bucket_name": bucket_name,
                    "source": source,
                }
        except (
            urllib3.exceptions.MaxRetryError,
            urllib3.exceptions.NewConnectionError,
            urllib3.exceptions.ConnectionError,
            RuntimeError,
        ) as e:
            print(str(e))
            raise Exception(e)

    def get_fn_args(self, input_dict: dict, exec_properties: dict):
        source = input_dict.get(standard_component_specs.MINIO_SOURCE)
        bucket_name = input_dict.get(standard_component_specs.MINIO_BUCKET_NAME)
        folder_name = input_dict.get(standard_component_specs.MINIO_BUCKET_NAME)
        endpoint = exec_properties.get(standard_component_specs.MINIO_ENDPOINT)
        return source, bucket_name, folder_name, endpoint

    def Do(self, input_dict: dict, output_dict: dict, exec_properties: dict):

        source, bucket_name, folder_name, endpoint = self.get_fn_args(
            input_dict=input_dict, exec_properties=exec_properties
        )

        minio_config = self._read_minio_creds(endpoint=endpoint)

        client = self._initiate_minio_client(minio_config=minio_config)

        if not os.path.exists(source):
            raise Exception("Input path - {} does not exists".format(source))

        if os.path.isfile(source):
            artifact_name = source.split("/")[-1]
            destination = os.path.join(folder_name, artifact_name)
            self.upload_artifacts_to_minio(
                client=client,
                source=source,
                destination=destination,
                bucket_name=bucket_name,
                output_dict=output_dict,
            )
        elif os.path.isdir(source):
            for root, dirs, files in os.walk(source):
                for file in files:
                    source = os.path.join(root, file)
                    artifact_name = source.split("/")[-1]
                    destination = os.path.join(folder_name, artifact_name)
                    self.upload_artifacts_to_minio(
                        client=client,
                        source=source,
                        destination=destination,
                        bucket_name=bucket_name,
                        output_dict=output_dict,
                    )
        else:
            raise ValueError("Unknown source: {} ".format(source))
