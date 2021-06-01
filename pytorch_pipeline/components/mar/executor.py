import os
import shutil
import wget
import tempfile
import subprocess
from pathlib import Path
from pytorch_pipeline.components.base.base_executor import BaseExecutor
from pytorch_pipeline.types import standard_component_specs


class Executor(BaseExecutor):
    def __init__(self):
        super(Executor, self).__init__()

    def get_fn_args(self, input_dict: dict, exec_properties: dict):

        mar_config = input_dict.get(standard_component_specs.MAR_GENERATION_CONFIG)
        mar_save_path = exec_properties.get(standard_component_specs.MAR_GENERATION_SAVE_PATH)
        return mar_config, mar_save_path

    def _validate_mar_config(self, mar_config):

        mandatory_args = [
            "MODEL_NAME",
            "SERIALIZED_FILE",
            "MODEL_FILE",
            "HANDLER",
            "VERSION",
            "CONFIG_PROPERTIES",
        ]

        if not mar_config:
            raise ValueError(
                f"Mar config cannot be empty. Mandatory arguments are {mandatory_args}"
            )

        missing_list = []
        for key in mandatory_args:
            if key not in mar_config:
                missing_list.append(key)

        if missing_list:
            raise ValueError(
                "Following Mandatory keys are missing in the config file {}".format(missing_list)
            )

    def download_config_properties(self, url):
        if not os.path.exists(url):
            url = wget.download(url, tempfile.mkdtemp())
        return url

    def _generate_mar_file(self, mar_config: dict, mar_save_path: str, output_dict: dict):

        self._validate_mar_config(mar_config=mar_config)

        for key, uri in mar_config.items():
            # uri = self._download_dependent_file(key, uri)
            mar_config[key] = uri

        archiver_cmd = (
            "torch-model-archiver --force "
            "--model-name {MODEL_NAME} "
            "--serialized-file {SERIALIZED_FILE} "
            "--model-file {MODEL_FILE} "
            "--handler {HANDLER} "
            "-v {VERSION}".format(
                MODEL_NAME=mar_config["MODEL_NAME"],
                SERIALIZED_FILE=mar_config["SERIALIZED_FILE"],
                MODEL_FILE=mar_config["MODEL_FILE"],
                HANDLER=mar_config["HANDLER"],
                VERSION=mar_config["VERSION"],
            )
        )

        if "EXPORT_PATH" in mar_config:
            export_path = mar_config["EXPORT_PATH"]
            output_dict[standard_component_specs.MAR_GENERATION_SAVE_PATH] = export_path
            if not os.path.exists(export_path):
                Path(export_path).mkdir(parents=True, exist_ok=True)

            archiver_cmd += " --export-path {EXPORT_PATH}".format(EXPORT_PATH=export_path)

        if "EXTRA_FILES" in mar_config:
            archiver_cmd += " --extra-files {EXTRA_FILES}".format(
                EXTRA_FILES=mar_config["EXTRA_FILES"]
            )

        if "REQUIREMENTS_FILE" in mar_config:
            archiver_cmd += " -r {REQUIREMENTS_FILE}".format(
                REQUIREMENTS_FILE=mar_config["REQUIREMENTS_FILE"]
            )

        print("Running Archiver cmd: ", archiver_cmd)

        return_code = subprocess.Popen(archiver_cmd, shell=True).wait()
        if return_code != 0:
            error_msg = "Error running command {archiver_cmd} {return_code}".format(
                archiver_cmd=archiver_cmd, return_code=return_code
            )
            print(error_msg)
            raise Exception("Unable to create mar file: {error_msg}".format(error_msg=error_msg))

        # If user has provided the export path
        # By default, torch-model-archiver generates the mar file inside the export path

        # If the user has not provieded the export path
        # mar file will be generated in the current working directory
        # The mar file needs to be moved into mar_save_path

        if "EXPORT_PATH" not in mar_config:
            mar_file_local_path = os.path.join(
                os.getcwd(), "{}.mar".format(mar_config["MODEL_NAME"])
            )
            if not Path(mar_save_path).exists():
                Path(mar_save_path).mkdir(parents=True, exist_ok=True)
            shutil.move(mar_file_local_path, mar_save_path)
            output_dict[standard_component_specs.MAR_GENERATION_SAVE_PATH] = mar_save_path

        elif mar_config["EXPORT_PATH"] != mar_save_path:
            raise Exception(
                "The export path [{}] needs to be same as mar save path [{}] ".format(
                    mar_config["EXPORT_PATH"], mar_save_path
                )
            )

        print("Saving model file ")
        ## TODO: While separating the mar generation component from trainer
        ## Create a separate url for model file
        print(f"copying {mar_config['MODEL_FILE']} to {mar_config['EXPORT_PATH']}")
        shutil.copy(mar_config["MODEL_FILE"], mar_config["EXPORT_PATH"])

    def _save_config_properties(self, mar_config: dict, mar_save_path: str, output_dict: dict):
        print("Downloading config properties")
        if "CONFIG_PROPERTIES" in mar_config:
            config_properties_local_path = self.download_config_properties(
                mar_config["CONFIG_PROPERTIES"]
            )
        else:
            config_properties_local_path = mar_config["CONFIG_PROPERTIES"]

        config_prop_path = os.path.join(mar_save_path, "config.properties")
        if os.path.exists(config_prop_path):
            os.remove(config_prop_path)
        shutil.move(config_properties_local_path, mar_save_path)
        output_dict[standard_component_specs.CONFIG_PROPERTIES_SAVE_PATH] = mar_save_path

    def Do(self, input_dict: dict, output_dict: dict, exec_properties: dict):
        self._log_startup(
            input_dict=input_dict, output_dict=output_dict, exec_properties=exec_properties
        )
        mar_config, mar_save_path = self.get_fn_args(
            input_dict=input_dict, exec_properties=exec_properties
        )
        self._validate_mar_config(mar_config=mar_config)

        self._generate_mar_file(
            mar_config=mar_config, mar_save_path=mar_save_path, output_dict=output_dict
        )
        self._save_config_properties(
            mar_config=mar_config, mar_save_path=mar_save_path, output_dict=output_dict
        )
