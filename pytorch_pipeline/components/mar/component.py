from pytorch_pipeline.components.mar.executor import Executor
from pytorch_pipeline.types import standard_component_specs
from pytorch_pipeline.components.base.base_component import BaseComponent


class MarGeneration(BaseComponent):
    def __init__(self, mar_config: dict, mar_save_path: str = None):
        super(MarGeneration, self).__init__()
        input_dict = {
            standard_component_specs.MAR_GENERATION_CONFIG: mar_config,
        }

        output_dict = {}

        exec_properties = {standard_component_specs.MAR_GENERATION_SAVE_PATH: mar_save_path}

        spec = standard_component_specs.MarGenerationSpec()
        self._validate_spec(
            spec=spec,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
        )

        Executor().Do(
            input_dict=input_dict, output_dict=output_dict, exec_properties=exec_properties
        )
        self.output_dict = output_dict
