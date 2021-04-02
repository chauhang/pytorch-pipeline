import pandas as pd
import json
import os

from sklearn.metrics import confusion_matrix


class Visualization:
    def __init__(self):
        self.parser_args = None

    def _generate_confusion_matrix_metadata(self, confusion_matrix_path):
        print("Generating Confusion matrix Metadata")
        metadata = {
            "outputs": [
                {
                    "type": "confusion_matrix",
                    "format": "csv",
                    "schema": [
                        {"name": "target", "type": "CATEGORY"},
                        {"name": "predicted", "type": "CATEGORY"},
                        {"name": "count", "type": "NUMBER"},
                    ],
                    "source": os.path.join(confusion_matrix_path, "confusion_matrix.csv"),
                    # Convert vocab to string because for bealean values we want "True|False" to match csv data.
                    "labels": "vocab",
                }
            ]
        }
        with open("/mlpipeline-ui-metadata.json", "w") as f:
            json.dump(metadata, f)

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

    def _enable_tensorboard_visualization(self, tensorboard_root):
        print("Enabling Tensorboard Visualization")
        metadata = {
            "type": "tensorboard",
            "source": tensorboard_root,
        }
        self._write_ui_metadata(
            metadata_filepath="/mlpipeline-ui-metadata.json", metadata_dict=metadata
        )

    def _visualize_accuracy_metric(self, accuracy):
        metadata = {
            "name": "accuracy-score",
            "numberValue": accuracy,
            "format": "PERCENTAGE",
        }
        self._write_ui_metadata(
            metadata_filepath="/mlpipeline-metrics.json", metadata_dict=metadata, key="metrics"
        )

    def generate_visualization(self, tensorboard_root=None, accuracy=None):
        print("Tensorboard Root: {}".format(tensorboard_root))
        print("Accuracy: {}".format(accuracy))

        if tensorboard_root:
            self._enable_tensorboard_visualization(tensorboard_root)

        if accuracy:
            self._visualize_accuracy_metric(accuracy=accuracy)

        # confusion_matrix_path = self.parser_args["confusion_matrix_path"]
        # print('Confusion Matrix Path: {}'.format(confusion_matrix_path))
        #
        # if confusion_matrix_path:
        #     self._generate_confusion_matrix_metadata(confusion_matrix_path=confusion_matrix_path)


def generate_confusion_matrix(actuals, preds, output_path):
    confusion_matrix_output = confusion_matrix(actuals, preds)
    confusion_df = pd.DataFrame(confusion_matrix_output)
    confusion_df.to_csv(output_path, index=False)
