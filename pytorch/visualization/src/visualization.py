import json
from argparse import ArgumentParser


class Visualization:
    def __init__(self):
        self.parser_args = None

    def _parser_input_arguments(self):
        parser = ArgumentParser(add_help=False)
        parser.add_argument(
            "--confusion_matrix_path",
            type=str,
            default="",
            help="Confusion Matrix path (default: '')",
        )

        self.parser_args = vars(parser.parse_args())

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
                    "source": confusion_matrix_path,
                    # Convert vocab to string because for bealean values we want "True|False" to match csv data.
                    "labels": "vocab",
                }
            ]
        }
        with open("/mlpipeline-ui-metadata.json", "w") as f:
            json.dump(metadata, f)

    def generate_visualization(self):
        self._parser_input_arguments()

        confusion_matrix_path = self.parser_args["confusion_matrix_path"]
        print('Confusion Matrix Path: {}'.format(confusion_matrix_path))

        if confusion_matrix_path:
            self._generate_confusion_matrix_metadata(confusion_matrix_path=confusion_matrix_path)


if __name__ == "__main__":
    Visualization().generate_visualization()
