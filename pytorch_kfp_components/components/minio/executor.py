import os
from pytorch_kfp_components.components.base.base_executor import BaseExecutor


class Executor(BaseExecutor):
    def __init__(self):
        super(Executor, self).__init__()

    def _upload_artifacts_to_minio(self, client, source, destination, bucket_name):
        print(f"source {source} destination {destination}")
        result = client.fput_object(
            bucket_name=bucket_name,
            file_path=source,
            object_name=destination,
        )
        print(result)

        # Validate the output

    def Do(self, client, config, source, bucket_name, folder_name):

        if os.path.isfile(source):
            artifact_name = source.split("/")[-1]
            destination = os.path.join(folder_name, artifact_name)
            self._upload_artifacts_to_minio(
                client=client, source=source, destination=destination, bucket_name=bucket_name
            )

        elif os.path.isdir(source):
            for root, dirs, files in os.walk(source):
                for file in files:
                    print("Path")
                    source = os.path.join(root, file)
                    artifact_name = source.split("/")[-1]
                    destination = os.path.join(folder_name, artifact_name)
                    self._upload_artifacts_to_minio(
                        client=client,
                        source=source,
                        destination=destination,
                        bucket_name=bucket_name,
                    )
