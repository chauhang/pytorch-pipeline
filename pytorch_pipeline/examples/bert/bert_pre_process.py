import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import pyarrow.csv as pv
import pyarrow.parquet as pq
from torchtext.utils import download_from_url, extract_archive

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_url",
        default="https://kubeflow-dataset.s3.us-east-2.amazonaws.com/ag_news_csv.tar.gz",
        type=str,
        help="URL to download AG News dataset",
    )

    parser.add_argument(
        "--output_path",
        default="output/processing",
        type=str,
        help="Path to write the ag news dataset",
    )

    args = vars(parser.parse_args())

    dataset_url = args["dataset_url"]
    output_path = args["output_path"]

    Path(output_path).mkdir(parents=True, exist_ok=True)

    dataset_tar = download_from_url(dataset_url, root="./")
    extracted_files = extract_archive(dataset_tar)

    ag_news_csv = pv.read_csv("ag_news_csv/train.csv")

    pq.write_table(ag_news_csv, os.path.join(output_path, "ag_news_data.parquet"))

    entry_point = ["ls", "-R", output_path]
    run_code = subprocess.run(entry_point, stdout=subprocess.PIPE)
    print(run_code.stdout)
