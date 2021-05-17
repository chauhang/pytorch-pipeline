import pandas as pd
import os
import json
import tempfile
from sklearn.metrics import confusion_matrix
from pytorch_pipeline.components.base.base_executor import BaseExecutor


class Executor(BaseExecutor):
    def __init__(self):
        pass

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

    def _generate_confusion_matrix_metadata(
        self, confusion_matrix_path, classes, mlpipeline_ui_metadata
    ):
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
        self._write_ui_metadata(metadata_filepath=mlpipeline_ui_metadata, metadata_dict=metadata)

    def _generate_confusion_matrix(self, confusion_matrix_dict, mlpipeline_ui_metadata):
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
            mlpipeline_ui_metadata=mlpipeline_ui_metadata,
        )

    def Do(
        self,
        mlpipeline_ui_metadata=None,
        mlpipeline_metrics=None,
        confusion_matrix_dict=None,
        test_accuracy=None,
    ):
        if confusion_matrix:
            self._generate_confusion_matrix(
                confusion_matrix_dict=confusion_matrix_dict,
                mlpipeline_ui_metadata=mlpipeline_ui_metadata,
            )
