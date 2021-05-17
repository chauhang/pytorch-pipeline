import pandas as pd
import os
import json
import tempfile
from sklearn.metrics import confusion_matrix
from pytorch_pipeline.components.base.base_executor import BaseExecutor


class Executor(BaseExecutor):
    def __init__(self, mlpipeline_ui_metadata, mlpipeline_metrics):
        self.mlpipeline_ui_metadata = mlpipeline_ui_metadata
        self.mlpipeline_metrics = mlpipeline_metrics

    def _write_ui_metadata(self, metadata_filepath, metadata_dict, key="outputs"):
        if not os.path.exists(metadata_filepath):
            metadata = {key: [metadata_dict]}
        else:
            with open(metadata_filepath) as fp:
                metadata = json.load(fp)
                metadata_outputs = metadata[key]
                metadata_outputs.append(metadata_dict)

        print("Writing to file: {}".format(metadata_filepath))
        with open(metadata_filepath, "w") as fp:
            json.dump(metadata, fp)

    def _generate_confusion_matrix_metadata(self, confusion_matrix_path, classes):
        print("Generating Confusion matrix Metadata")
        metadata = {
            "type": "confusion_matrix",
            "format": "csv",
            "schema": [
                {"name": "target", "type": "CATEGORY"},
                {"name": "predicted", "type": "CATEGORY"},
                {"name": "count", "type": "NUMBER"},
            ],
            "source": confusion_matrix_path,
            "labels": list(map(str, classes)),
        }
        self._write_ui_metadata(
            metadata_filepath=self.mlpipeline_ui_metadata, metadata_dict=metadata
        )

    def _generate_confusion_matrix(self, confusion_matrix_dict):
        actuals = confusion_matrix_dict["actuals"]
        preds = confusion_matrix_dict["preds"]

        # Generating confusion matrix
        df = pd.DataFrame(list(zip(actuals, preds)), columns=["target", "predicted"])
        vocab = list(df["target"].unique())
        cm = confusion_matrix(df["target"], df["predicted"], labels=vocab)
        data = []
        for target_index, target_row in enumerate(cm):
            for predicted_index, count in enumerate(target_row):
                data.append((vocab[target_index], vocab[predicted_index], count))

        confusion_matrix_df = pd.DataFrame(data, columns=["target", "predicted", "count"])

        confusion_matrix_output_dir = str(tempfile.mkdtemp())
        confusion_matrix_output_path = os.path.join(
            confusion_matrix_output_dir, "confusion_matrix.csv"
        )
        # saving confusion matrix
        confusion_matrix_df.to_csv(confusion_matrix_output_path, index=False, header=False)

        # Generating metadata
        self._generate_confusion_matrix_metadata(
            confusion_matrix_path=confusion_matrix_output_path,
            classes=vocab,
        )

    def _visualize_accuracy_metric(self, accuracy):
        metadata = {
            "name": "accuracy-score",
            "numberValue": accuracy,
            "format": "PERCENTAGE",
        }
        self._write_ui_metadata(
            metadata_filepath=self.mlpipeline_metrics, metadata_dict=metadata, key="metrics"
        )

    def Do(
        self,
        confusion_matrix_dict=None,
        test_accuracy=None,
    ):
        if confusion_matrix:
            self._generate_confusion_matrix(
                confusion_matrix_dict=confusion_matrix_dict,
            )

        if test_accuracy:
            self._visualize_accuracy_metric(accuracy=test_accuracy)
