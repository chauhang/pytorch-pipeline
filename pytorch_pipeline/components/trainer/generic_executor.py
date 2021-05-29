import importlib
import inspect
from pytorch_pipeline.components.base.base_executor import BaseExecutor
from pytorch_pipeline.types import standard_component_specs


class GenericExecutor(BaseExecutor):
    def Do(self, input_dict: dict, output_dict: dict, exec_properties: dict):
        # TODO: Code to train pretrained model
        pass

    def _GetFnArgs(self, input_dict: dict, output_dict: dict, execution_properties: dict):
        module_file = input_dict.get(standard_component_specs.TRAINER_MODULE_CLASS)
        data_module_file = input_dict.get(standard_component_specs.TRAINER_DATA_MODULE_CLASS)
        trainer_args = execution_properties.get(standard_component_specs.PTL_TRAINER_ARGS)
        module_file_args = execution_properties.get(standard_component_specs.TRAINER_MODULE_ARGS)
        data_module_args = execution_properties.get(
            (standard_component_specs.TRAINER_DATA_MODULE_ARGS)
        )
        return module_file, data_module_file, trainer_args, module_file_args, data_module_args

    def derive_model_and_data_module_class(self, module_file: str, data_module_file: str):
        model_class = None
        data_module_class = None

        class_module = importlib.import_module(module_file.split(".")[0])
        data_module = importlib.import_module(data_module_file.split(".")[0])

        for cls in inspect.getmembers(
            class_module,
            lambda member: inspect.isclass(member) and member.__module__ == class_module.__name__,
        ):
            model_class = cls[1]

        if not model_class:
            raise ValueError(f"Unable to load module_file - {module_file}")

        for cls in inspect.getmembers(
            data_module,
            lambda member: inspect.isclass(member) and member.__module__ == data_module.__name__,
        ):
            data_module_class = cls[1]

        if not data_module_class:
            raise ValueError(f"Unable to load data_module_file - {data_module_file}")

        return model_class, data_module_class
