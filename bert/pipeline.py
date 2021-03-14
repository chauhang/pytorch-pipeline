import kfp
import json
import os
import copy
from kfp import components
from kfp import dsl
from kfp.aws import use_aws_secret
from kfp.components import load_component_from_file, load_component_from_url


cur_file_dir = os.path.dirname(__file__)
components_dir = os.path.join(cur_file_dir, "../pytorch")

bert_data_prep_op = components.load_component_from_file(
    components_dir + "/data_prep/component.yaml"
)

bert_train_op = components.load_component_from_file(
    components_dir + "/train/component.yaml"
)

deploy_op = load_component_from_url("https://raw.githubusercontent.com/kubeflow/pipelines/97eae83a96b0ac87805e2d6db6097e479bb38b1f/components/kubeflow/kfserving/component.yaml")

@dsl.pipeline(name="Training pipeline", description="Sample training job test")
def training(input_directory = "/pvc/input",
    output_directory = "/pvc/output", handlerFile = "image_classifier"):

    namespace = "admin"
    volume_name = "pvcm"

    vop = dsl.VolumeOp(
        name=volume_name,
        resource_name=volume_name,
        modes=dsl.VOLUME_MODE_RWO,
        size="5Gi"
    )

    @dsl.component
    def download(url: str, output_path:str):
        return dsl.ContainerOp(
            name='Download',
            image='busybox:latest',
            command=["sh", "-c"],
            arguments=["mkdir -p %s; wget %s -P %s" % (output_path, url, output_path)],
        )

    @dsl.component
    def copy_contents(input_dir: str, output_dir:str):
        return dsl.ContainerOp(
            name='Download',
            image='busybox:latest',
            command=["cp", "-R", "%s/." % input_dir, "%s" % output_dir],
        )

    @dsl.component
    def ls(input_dir: str):
        return dsl.ContainerOp(
            name='list',
            image='busybox:latest',
            command=["ls", "-R", "%s" % input_dir]
        )

    prep_output = bert_data_prep_op(
        input_data =
            [{"dataset_url":"https://kubeflow-dataset.s3.us-east-2.amazonaws.com/ag_news_csv.tar.gz"}],
        container_entrypoint = [
            "python",
            "/pvc/input/bert_pre_process.py",
        ],
        output_data = ["/pvc/output/processing"],
        source_code = ["https://kubeflow-dataset.s3.us-east-2.amazonaws.com/bert_pre_process.py"],
        source_code_path = ["/pvc/input"]
    ).add_pvolumes({"/pvc":vop.volume})

    train_output = bert_train_op(
        input_data = ["/pvc/output/processing"],
        container_entrypoint = [
            "python",
            "/pvc/input/bert_train.py",
        ],
        output_data = ["/pvc/output/train/models"],
        input_parameters = [{"tensorboard_root": "/pvc/output/train/tensorboard", 
        "max_epochs": 1, "num_samples": 150, "batch_size": 4, "num_workers": 1, "learning_rate": 0.001, 
        "accelerator": None}],
        source_code = ["https://kubeflow-dataset.s3.us-east-2.amazonaws.com/bert_datamodule.py", "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/bert_train.py"],
        source_code_path = ["/pvc/input"]
    ).add_pvolumes({"/pvc":vop.volume}).after(prep_output)

    list_input = ls("/pvc/output").add_pvolumes({"/pvc":vop.volume}).after(train_output)

    properties = download(url='https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/bert/properties.json', output_path="/pv/input").add_pvolumes({"/pv":vop.volume}).after(vop)
    requirements = download(url='https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/bert/requirements.txt', output_path="/pv/input").add_pvolumes({"/pv":vop.volume}).after(vop)
    extrafile = download(url='https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/bert/index_to_name.json', output_path="/pv/input").add_pvolumes({"/pv":vop.volume}).after(vop)
    vocabfile = download(url='https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/bert/source_vocab.pt', output_path="/pv/input").add_pvolumes({"/pv":vop.volume}).after(vop)

    copy_files = copy_contents(input_dir="/pvc/output/train/models", output_dir="/pvc/input").add_pvolumes({"/pvc":vop.volume}).after(train_output)
    list_input = ls("/pvc/input").add_pvolumes({"/pvc":vop.volume}).after(copy_files)

    mar_task = dsl.ContainerOp(
        name='mar_gen',
        image='jagadeeshj/model_archive_step:kfpv1.2',
        command=["/usr/local/bin/dockerd-entrypoint.sh"],
        arguments=[
            "--output_path", output_directory,
            "--input_path", input_directory
        ],
        pvolumes={"/pvc": vop.volume}).after(list_input)

    list_output = ls("/pvc/output").add_pvolumes({"/pvc":vop.volume}).after(mar_task)

    deploy = deploy_op(
        action="create", 
        model_name="torchserve-bert", 
        model_uri="pvc://{{workflow.name}}-pvcm/output", 
        namespace='admin',
        framework='pytorch'
    ).after(list_output)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(training, package_path="pytorch_bert.yaml")