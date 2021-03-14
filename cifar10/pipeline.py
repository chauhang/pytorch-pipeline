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

cifar10_data_prep_op = components.load_component_from_file(
    components_dir + "/data_prep/component.yaml"
)

cifar10_train_op = components.load_component_from_file(
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
        size="1Gi"
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

    prep_output = cifar10_data_prep_op(
        input_data =
            [],
        container_entrypoint = [
            "python",
            "/pvc/input/cifar10_pre_process.py",
        ],
        output_data = ["/pvc/output/processing"],
        source_code = ["https://kubeflow-dataset.s3.us-east-2.amazonaws.com/cifar10_pre_process.py"],
        source_code_path = ["/pvc/input"]
    ).add_pvolumes({"/pvc":vop.volume})

    train_output = cifar10_train_op(
        input_data = ["/pvc/output/processing"],
        container_entrypoint = [
            "python",
            "/pvc/input/cifar10_train.py",
        ],
        output_data = ["/pvc/output/train/models"],
        input_parameters = [{"tensorboard_root": "/pvc/output/train/tensorboard", 
        "max_epochs": 1, "gpus": 0, "train_batch_size": None, "val_batch_size": None, "train_num_workers": 4, 
        "val_num_workers": 4 , "learning_rate": 0.001, 
        "accelerator": None}],
        source_code = ["https://kubeflow-dataset.s3.us-east-2.amazonaws.com/cifar10_datamodule.py", "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/cifar10_train.py"],
        source_code_path = ["/pvc/input"]
    ).add_pvolumes({"/pvc":vop.volume}).after(prep_output)

    list_input = ls("/pvc/output").add_pvolumes({"/pvc":vop.volume}).after(train_output)

    properties = download(url='https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/properties.json', output_path="/pv/input").add_pvolumes({"/pv":vop.volume}).after(vop)
    index_to_name = download(url='https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/index_to_name.json', output_path="/pv/input").add_pvolumes({"/pv":vop.volume}).after(vop)
    requirements = download(url='https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/requirements.txt', output_path="/pv/input").add_pvolumes({"/pv":vop.volume}).after(vop)

    copy_files = copy_contents(input_dir="/pvc/output/train/models", output_dir="/pvc/input").add_pvolumes({"/pvc":vop.volume}).after(train_output)
    list_input = ls("/pvc/input").add_pvolumes({"/pvc":vop.volume}).after(copy_files)

    mar_task = dsl.ContainerOp(
        name='mar_gen',
        image='jagadeeshj/model_archive_step:kfpv1.2',
        command=["/usr/local/bin/dockerd-entrypoint.sh"],
        arguments=[
            "--output_path", output_directory,
            "--input_path", input_directory,
            "--handlerfile", handlerFile
        ],
        pvolumes={"/pvc": vop.volume}).after(list_input)

    list_output = ls("/pvc/output").add_pvolumes({"/pvc":vop.volume}).after(mar_task)

    deploy = deploy_op(
        action="create", 
        model_name="torchserve", 
        model_uri="pvc://{{workflow.name}}-pvcm/output", 
        namespace='admin',
        framework='pytorch'
    ).after(list_output)

    # op_task = dsl.ContainerOp(
    #     name='main',
    #     image="quay.io/aipipeline/kfserving-component:v0.5.0",
    #     command=['python'],
    #     arguments=[
    #       "-u", "kfservingdeployer.py",
    #       "--action", "apply",
    #       "--model-name", "torchserve",
    #       "--model-uri", "pvc://{{workflow.name}}-%s/input" % volume_name,
    #       "--namespace", "%s" % namespace,
    #       "--framework", "pytorch",
    #     ],
    #     pvolumes={"/pvc": vop.volume},
    #     # pass in init_container list
    #     init_containers=[
    #         dsl.UserContainer(
    #             name='init',
    #             image='jagadeeshj/model_archive_step:kfpv1.0',
    #             command=["/usr/local/bin/dockerd-entrypoint.sh"],
    #             args=[
    #                 "--output_path", output_directory,
    #                 "--input_path", input_directory,
    #                 "--handlerfile", handlerFile
    #             ],
    #             mirror_volume_mounts=True,),
    #     ],
    # ).after(list_input)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(training, package_path="pytorch_cifar10.yaml")