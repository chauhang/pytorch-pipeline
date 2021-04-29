import os
from minio import Minio


class LibMinio:
    def __init__(self, minio_config: dict = None):
        if not minio_config:
            self.minio_config = self._use_default_config()
        else:
            self.minio_config = minio_config
        self.client = self._initiate_minio_client()

    def _use_default_config(self):
        print("Using Default minio config")
        minio_config = {
            "HOST": "minio-service.kubeflow:9000",
            "ACCESS_KEY": "minio",
            "SECRET_KEY": "minio123",
            "BUCKET": "mlpipeline",
            "FOLDER": "mar",
        }

        return minio_config

    def _initiate_minio_client(self):
        minio_host = self.minio_config["HOST"]
        access_key = self.minio_config["ACCESS_KEY"]
        secret_key = self.minio_config["SECRET_KEY"]
        client = Minio(minio_host, access_key=access_key, secret_key=secret_key, secure=False)
        return client

    def upload_artifact_to_minio(self, folder: str, artifact: str):
        artifact_name = artifact.split("/")[-1]
        result = self.client.fput_object(
            self.minio_config["BUCKET"],
            os.path.join(folder, artifact_name),
            artifact,
        )
        print(result)


# if __name__ == "__main__":
#     minio_config = {
#         "HOST": "172.17.0.2:9000",
#         "ACCESS_KEY": "minioadmin",
#         "SECRET_KEY": "minioadmin",
#         "BUCKET": "kubeflow-dataset",
#         "FOLDER": "mar",
#     }
#
#     LibMinio(minio_config=minio_config).upload_artifact_to_minio("output/train/models/resnet.pth")
