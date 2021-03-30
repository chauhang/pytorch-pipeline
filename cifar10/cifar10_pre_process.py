import subprocess
from argparse import ArgumentParser
from pathlib import Path

import torchvision
import webdataset as wds
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--output_path",
        type=str,
        default="/pvc/output/processing",
        help="Output path to download cifar 10 dataset (default: /pvc/output/processing)",
    )

    args = vars(parser.parse_args())

    output_path = args["output_path"]

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
