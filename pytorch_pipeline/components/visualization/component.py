import os
from pathlib import Path
from pytorch_pipeline.components.base.base_component import BaseComponent
from pytorch_pipeline.components.visualization.Executor import Executor


class Visualization(BaseComponent):
    def __init__(
        self,
        mlpipeline_ui_metadata=None,
        mlpipeline_metrics=None,
        pod_template_spec=None,
        confusion_matrix_dict=None,
        test_accuracy=None,
    ):
        super(BaseComponent, self).__init__()

        if mlpipeline_ui_metadata:
            Path(os.path.dirname(mlpipeline_ui_metadata)).mkdir(parents=True, exist_ok=True)
        else:
            mlpipeline_ui_metadata = "/mlpipeline-ui-metadata.json"

        if mlpipeline_metrics:
            Path(os.path.dirname(mlpipeline_metrics)).mkdir(parents=True, exist_ok=True)
        else:
            mlpipeline_metrics = "/mlpipeline-metrics.json"

        Executor(
            mlpipeline_ui_metadata=mlpipeline_ui_metadata,
            mlpipeline_metrics=mlpipeline_metrics,
            pod_template_spec=pod_template_spec,
        ).Do(
            confusion_matrix_dict=confusion_matrix_dict,
            test_accuracy=test_accuracy,
        )
