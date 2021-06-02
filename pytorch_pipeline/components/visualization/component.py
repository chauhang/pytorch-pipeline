from pytorch_pipeline.components.base.base_component import BaseComponent
from pytorch_pipeline.components.visualization.executor import Executor
from pytorch_pipeline.types import standard_component_specs


class Visualization(BaseComponent):
    def __init__(
        self,
        mlpipeline_ui_metadata=None,
        mlpipeline_metrics=None,
        confusion_matrix_dict=None,
        test_accuracy=None,
        markdown=None,
    ):
        super(BaseComponent, self).__init__()

        input_dict = {
            standard_component_specs.VIZ_CONFUSION_MATRIX_DICT: confusion_matrix_dict,
            standard_component_specs.VIZ_TEST_ACCURACY: test_accuracy,
            standard_component_specs.VIZ_MARKDOWN: markdown,
        }

        output_dict = {}

        exec_properties = {
            standard_component_specs.VIZ_MLPIPELINE_UI_METADATA: mlpipeline_ui_metadata,
            standard_component_specs.VIZ_MLPIPELINE_METRICS: mlpipeline_metrics,
        }

        spec = standard_component_specs.VisualizationSpec()
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
