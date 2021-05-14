import os
from pytorch_pipeline.components.base.base_component import BaseComponent
from pytorch_pipeline.components.minio.executor import Executor
from minio import Minio


class MinIO(BaseComponent):
    def __init__(
        self, source: str, bucket_name, destination, minio_config: dict = None, endpoint=None
    ):
        super(BaseComponent, self).__init__()

        self.endpoint = endpoint

        if not minio_config:
            self.minio_config = self._use_default_config()
        else:
            self.minio_config = minio_config
        self.client = self._initiate_minio_client()

        if os.path.exists(source):
            Executor().Do(self.client, self.minio_config, source, bucket_name, destination)
        else:
            raise Exception("Input path - {} does not exists".format(source))

    def _use_default_config(self):
        # TODO: read the default config from secret manager
        print("Using Default minio config")
        minio_config = {
            "HOST": self.endpoint,
            "ACCESS_KEY": os.environ["MINIO_ACCESS_KEY"],
            "SECRET_KEY": os.environ["MINIO_SECRET_KEY"],
        }

        return minio_config

    def _initiate_minio_client(self):
        minio_host = self.minio_config["HOST"]
        access_key = self.minio_config["ACCESS_KEY"]
        secret_key = self.minio_config["SECRET_KEY"]
        client = Minio(minio_host, access_key=access_key, secret_key=secret_key, secure=False)
        return client
