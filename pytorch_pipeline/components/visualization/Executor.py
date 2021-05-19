import pandas as pd
import os
import json
import tempfile
import boto3
from io import StringIO
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix
from pytorch_pipeline.components.base.base_executor import BaseExecutor
from pytorch_pipeline.components.minio.component import MinIO


class Executor(BaseExecutor):
    def __init__(self, mlpipeline_ui_metadata, mlpipeline_metrics, pod_template_spec=None):
        self.mlpipeline_ui_metadata = mlpipeline_ui_metadata
        self.mlpipeline_metrics = mlpipeline_metrics
        self.pod_template_spec = pod_template_spec

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

    def _generate_markdown(self, markdown_dict):

        ##### Plain Markdonw
        # markdown_metadata = {
        #     "storage": markdown_dict["storage"],
        #     "source": json.dumps(markdown_dict["source"]),
        #     "type": "markdown",
        # }

        ###### Web APP
        # source_str = json.dumps(markdown_dict["source"])
        # source = f"<font size='5'> {source_str} <font/>"
        # markdown_metadata = {
        #     "storage": markdown_dict["storage"],
        #     "source": source,
        #     "type": "web-app",
        # }

        source_str = json.dumps(markdown_dict["source"], sort_keys=True, indent=4)
        source = f"```json \n {source_str} ```"
        markdown_metadata = {
            "storage": markdown_dict["storage"],
            "source": source,
            "type": "markdown",
        }


        self._write_ui_metadata(
            metadata_filepath=self.mlpipeline_ui_metadata, metadata_dict=markdown_metadata
        )

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

        if self.pod_template_spec:
            metadata["pod_template_spec"] = self.pod_template_spec

        self._write_ui_metadata(
            metadata_filepath=self.mlpipeline_ui_metadata, metadata_dict=metadata
        )

    def _generate_confusion_matrix(self, confusion_matrix_dict):
        actuals = confusion_matrix_dict["actuals"]
        preds = confusion_matrix_dict["preds"]
        confusion_matrix_url = confusion_matrix_dict["url"]

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
        #saving confusion matrix
        confusion_matrix_df.to_csv(confusion_matrix_output_path, index=False, header=False)

        parse_obj = urlparse(confusion_matrix_url, allow_fragments=False)
        bucket_name = parse_obj.netloc
        folder_name = str(parse_obj.path).lstrip("/")
        # confusion_matrix_key = os.path.join(folder_name, "confusion_matrix.csv")

        print("Bucket name: ", bucket_name)
        print("Folder name: ", folder_name)

        # csv_buffer = StringIO()
        # confusion_matrix_df.to_csv(csv_buffer, index=False, header=False)
        # s3_resource = boto3.resource("s3")
        #
        # s3_resource.Object(bucket_name, confusion_matrix_key).put(Body=csv_buffer.getvalue())
        # TODO:
        endpoint = "minio-service.kubeflow:9000"
        MinIO(
            source=confusion_matrix_output_path,
            bucket_name=bucket_name,
            destination=folder_name,
            endpoint=endpoint,
        )

        # Generating metadata
        self._generate_confusion_matrix_metadata(
            confusion_matrix_path=os.path.join(confusion_matrix_url, "confusion_matrix.csv"),
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

    def Do(self, confusion_matrix_dict=None, test_accuracy=None, markdown=None):
        if confusion_matrix_dict:
            self._generate_confusion_matrix(
                confusion_matrix_dict=confusion_matrix_dict,
            )

        if test_accuracy:
            self._visualize_accuracy_metric(accuracy=test_accuracy)

        if markdown:
            self._generate_markdown(markdown_dict=markdown)
