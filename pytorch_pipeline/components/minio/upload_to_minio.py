import os
from argparse import ArgumentParser
from pytorch_pipeline.components.minio.component import MinIO

# Argument parser for user defined paths
parser = ArgumentParser()

parser.add_argument(
    "--bucket_name",
    type=str,
    help="Minio bucket name",
)

parser.add_argument(
    "--folder_name",
    type=str,
    help="Path to destination folder",
)

parser.add_argument(
    "--input_path",
    type=str,
    help="Input path of the file or folder to upload",
)

parser.add_argument(
    "--filename",
    type=str,
    help="Name of the file to be uploaded",
)

parser.add_argument(
    "--endpoint",
    type=str,
    default="minio-service.kubeflow:9000",
    help="Name of the file to be uploaded",
)

args = vars(parser.parse_args())

bucket_name = args["bucket_name"]
input_path = args["input_path"]
folder_name = args["folder_name"]
filename = args["filename"]

if filename:
    input_path = os.path.join(input_path, filename)

endpoint = args["endpoint"]

print("File to be uploaded: {}".format(input_path))

print("Uploading file to : {}".format(folder_name))

MinIO(source=input_path, bucket_name=bucket_name, destination=folder_name, endpoint=endpoint)
