from pytorch_pipeline.components.base.base_component import BaseComponent
from pytorch_pipeline.components.visualization.Executor import Executor


class Visualization(BaseComponent):
    def __init__(
        self,
        mlpipeline_ui_metadata=None,
        mlpipeline_metrics=None,
        confusion_matrix_dict=None,
        test_accuracy=None,
    ):
        super(BaseComponent, self).__init__()

        if not mlpipeline_ui_metadata:
            mlpipeline_ui_metadata = "/mlpipeline-ui-metadata.json"

        if not mlpipeline_metrics:
            mlpipeline_metrics = "/mlpipeline-metrics.json"

        Executor(
            mlpipeline_ui_metadata=mlpipeline_ui_metadata, mlpipeline_metrics=mlpipeline_metrics
        ).Do(
            confusion_matrix_dict=confusion_matrix_dict,
            test_accuracy=test_accuracy,
        )
