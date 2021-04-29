import os
import wget
import tempfile
import subprocess
from pytorch_pipeline.components.utils.lib_minio import LibMinio


class MarGeneration:
    def __init__(self, mar_config: dict, minio_config: dict = None):
        self.mar_config = mar_config
        self.tmp_dirpath = tempfile.mkdtemp()
        self.minio_client = LibMinio(minio_config=minio_config)

    def _download_dependent_file(self, name, path):

        if name in ["MODEL_NAME", "VERSION"]:
            return path

        # Adding condition to use predefined torchserve handlers
        if name == "HANDLER" and path in [
            "image_classifier",
            "text_classifier",
            "image_segmenter",
            "object_detector",
        ]:
            return path

        if not os.path.exists(path):
            # TODO: Find a better way to check if a path is valid Uri
            print("Downloading {name} from url {path}".format(name=name, path=path))
            path = wget.download(path, self.tmp_dirpath)
        return path

    def _validate_mar_config(self):

        mandatory_args = [
            "MODEL_NAME",
            "SERIALIZED_FILE",
            "MODEL_FILE",
            "HANDLER",
            "VERSION",
            "CONFIG_PROPERTIES",
        ]

        missing_list = []
        for key in mandatory_args:
            if key not in self.mar_config:
                missing_list.append(key)

        if missing_list:
            raise Exception(
                "Following Mandatory keys are missing in the config file {} ".format(missing_list)
            )

    def download_config_properties(self, url):
        if not os.path.exists(url):
            url = wget.download(url, self.tmp_dirpath)
        return url

    def generate_mar_file(self):

        self._validate_mar_config()

        for key, uri in self.mar_config.items():
            # uri = self._download_dependent_file(key, uri)
            self.mar_config[key] = uri

        archiver_cmd = "torch-model-archiver --force --model-name {MODEL_NAME} --serialized-file {SERIALIZED_FILE} --model-file {MODEL_FILE} --handler {HANDLER} -v {VERSION}".format(
            MODEL_NAME=self.mar_config["MODEL_NAME"],
            SERIALIZED_FILE=self.mar_config["SERIALIZED_FILE"],
            MODEL_FILE=self.mar_config["MODEL_FILE"],
            HANDLER=self.mar_config["HANDLER"],
            VERSION=self.mar_config["VERSION"],
        )

        if "EXPORT_PATH" in self.mar_config:
            archiver_cmd += " --export-path {EXPORT_PATH}".format(
                EXPORT_PATH=self.mar_config["EXPORT_PATH"]
            )

        if "EXTRA_FILES" in self.mar_config:
            archiver_cmd += " --extra_files {EXTRA_FILES}".format(
                EXTRA_FILES=self.mar_config["EXTRA_FILES"]
            )

        if "REQUIREMENTS_FILE" in self.mar_config:
            archiver_cmd += " -r {REQUIREMENTS_FILE}".format(
                REQUIREMENTS_FILE=self.mar_config["REQUIREMENTS_FILE"]
            )

        print("Running Archiver cmd: ", archiver_cmd)

        return_code = subprocess.Popen(archiver_cmd, shell=True).wait()
        if return_code != 0:
            error_msg = "Error running command {archiver_cmd} {return_code}".format(
                archiver_cmd=archiver_cmd, return_code=return_code
            )
            print(error_msg)
            raise Exception("Unable to create mar file: {error_msg}".format(error_msg=error_msg))

        print("Downloading config properties")
        if "CONFIG_PROPERTIES" in self.mar_config:
            config_properties_local_path = self.download_config_properties(
                self.mar_config["CONFIG_PROPERTIES"]
            )
        else:
            config_properties_local_path = self.mar_config["CONFIG_PROPERTIES"]

        if "EXPORT_PATH" in self.mar_config:
            mar_file_local_path = os.path.join(
                self.mar_config["EXPORT_PATH"], "{}.mar".format(self.mar_config["MODEL_NAME"])
            )
        else:
            mar_file_local_path = os.path.join(
                os.getcwd(), "{}.mar".format(self.mar_config["MODEL_NAME"])
            )

        print("Uploading config properties to minio")
        self.minio_client.upload_artifact_to_minio(
            folder="mar/config", artifact=config_properties_local_path
        )

        print("Uploading mar file to minio")
        self.minio_client.upload_artifact_to_minio(
            folder="mar/model-store", artifact=mar_file_local_path
        )


# if __name__ == "__main__":
#     mar_config = {
#         "MODEL_NAME": "cifar10_test",
#         "MODEL_FILE": "examples/cifar10/cifar10_train.py",
#         "HANDLER": "image_classifier",
#         "SERIALIZED_FILE": "examples/cifar10/output/train/models/resnet.pth",
#         "VERSION": "2",
#         "CONFIG_PROPERTIES": "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/config.properties"
#     }
#     MarGeneration().generate_mar_file()
