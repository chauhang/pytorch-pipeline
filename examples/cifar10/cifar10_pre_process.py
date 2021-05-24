import subprocess
import numpy as np
from pathlib import Path

import torchvision
import webdataset as wds
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from pytorch_pipeline.components.visualization.component import Visualization

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_data", type=str)

    parser.add_argument(
        "--mlpipeline_ui_metadata",
        type=str,
        help="Path to write mlpipeline-ui-metadata.json",
    )

    args = vars(parser.parse_args())
    output_path = args["output_data"]

    Path(output_path).mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root="./", train=False, download=True)

    Path(output_path + "/train").mkdir(parents=True, exist_ok=True)
    Path(output_path + "/val").mkdir(parents=True, exist_ok=True)
    Path(output_path + "/test").mkdir(parents=True, exist_ok=True)

    random_seed = 25
    y = trainset.targets
    trainset, valset, y_train, y_val = train_test_split(
        trainset, y, stratify=y, shuffle=True, test_size=0.2, random_state=random_seed
    )

    for name in [(trainset, "train"), (valset, "val"), (testset, "test")]:
        with wds.ShardWriter(
            output_path + "/" + str(name[1]) + "/" + str(name[1]) + "-%d.tar", maxcount=1000
        ) as sink:
            for index, (image, cls) in enumerate(name[0]):
                sink.write({"__key__": "%06d" % index, "ppm": image, "cls": cls})

    entry_point = ["ls", "-R", output_path]
    run_code = subprocess.run(entry_point, stdout=subprocess.PIPE)
    print(run_code.stdout)

    visualization_arguments = {
        "output": {
            "mlpipeline_ui_metadata": args["mlpipeline_ui_metadata"],
            "dataset_download_path": args["output_data"],
        },
    }

    markdown_dict = {"storage": "inline", "source": visualization_arguments}

    print("Visualization arguments: ", markdown_dict)

    visualization = Visualization(
        mlpipeline_ui_metadata=args["mlpipeline_ui_metadata"],
        markdown=markdown_dict,
    )

    y_array = np.array(y)

    label_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    label_counts = dict(zip(*np.unique(y_array, return_counts=True)))
    label_dict = {}
    total_count = len(y)
    for key, value in label_counts.items():
        print("Label Counts of [{}]({}) : {}".format(key, label_names[key].upper(), value))
        label_dict[label_names[key].upper()] = int(value)

    label_dict["TOTAL_COUNT"] = int(total_count)

    markdown_dict = {"storage": "inline", "source": label_dict}

    visualization = Visualization(
        mlpipeline_ui_metadata=args["mlpipeline_ui_metadata"],
        markdown=markdown_dict,
    )