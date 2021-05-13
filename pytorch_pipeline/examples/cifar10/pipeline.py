import kfp
import json
from kfp import components
from kfp.components import load_component_from_file
from kfp import dsl
from kfp import compiler


DEPLOY = "torchserve"
MODEL = "cifar10"
namespace = "kubeflow-user-example-com"

yaml_folder_path = "pytorch_pipeline/examples/cifar10/yaml"

prepare_tensorboard_op = load_component_from_file(f"{yaml_folder_path}/tensorboard/component.yaml")
prep_op = components.load_component_from_file(f"{yaml_folder_path}/pre_process/component.yaml")
train_op = components.load_component_from_file(f"{yaml_folder_path}/train/component.yaml")
deploy_op = load_component_from_file(f"{yaml_folder_path}/deploy/component.yaml")


minio_op = components.load_component_from_file(f"{yaml_folder_path}/minio/component.yaml")


@dsl.pipeline(name="Training Cifar10 pipeline", description="Cifar 10 dataset pipeline")
def pytorch_cifar10(
    minio_endpoint="http://minio-service.kubeflow:9000",
    log_bucket="mlpipeline",
    log_dir=f"tensorboard/logs/{dsl.RUN_ID_PLACEHOLDER}/",
    mar_path=f"mar/{dsl.RUN_ID_PLACEHOLDER}/model-store",
    config_prop_path=f"mar/{dsl.RUN_ID_PLACEHOLDER}/config",
    model_uri=f"s3://mlpipeline/mar/{dsl.RUN_ID_PLACEHOLDER}",
    tf_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-3:latest",
):
    @dsl.component
    def ls(input_dir: str):
        return dsl.ContainerOp(
            name="list", image="busybox:latest", command=["ls", "-R", "%s" % input_dir]
        )

    prepare_tb_task = prepare_tensorboard_op(
        log_dir_uri=f"s3://{log_bucket}/{log_dir}",
        image=tf_image,
        pod_template_spec=json.dumps(
            {
                "spec": {
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "AWS_ACCESS_KEY_ID",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "mlpipeline-minio-artifact",
                                            "key": "accesskey",
                                        }
                                    },
                                },
                                {
                                    "name": "AWS_SECRET_ACCESS_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "mlpipeline-minio-artifact",
                                            "key": "secretkey",
                                        }
                                    },
                                },
                                {"name": "AWS_REGION", "value": "minio"},
                                {
                                    "name": "S3_ENDPOINT",
                                    "value": f"{minio_endpoint}",
                                },
                                {
                                    "name": "S3_USE_HTTPS",
                                    "value": "0",
                                },
                                {
                                    "name": "S3_VERIFY_SSL",
                                    "value": "0",
                                },
                            ]
                        }
                    ],
                },
            }
        ),
    ).set_display_name("Visualization")

    prep_task = prep_op().after(prepare_tb_task).set_display_name("Preprocess & Transform")
    train_task = (
        train_op(input_data=prep_task.outputs["output_data"])
        .after(prep_task)
        .set_display_name("Training")
    )
    minio_tb_upload = (
        minio_op(
            bucket_name="mlpipeline",
            folder_name=log_dir,
            input_path=train_task.outputs["tensorboard_root"],
            filename="",
        )
        .after(train_task)
        .set_display_name("Tensorboard Events Pusher")
    )
    minio_mar_upload = (
        minio_op(
            bucket_name="mlpipeline",
            folder_name=mar_path,
            input_path=train_task.outputs["checkpoint_dir"],
            filename="cifar10_test.mar",
        )
        .after(train_task)
        .set_display_name("Mar Pusher")
    )
    minio_config_upload = (
        minio_op(
            bucket_name="mlpipeline",
            folder_name=config_prop_path,
            input_path=train_task.outputs["checkpoint_dir"],
            filename="config.properties",
        )
        .after(train_task)
        .set_display_name("Conifg Pusher")
    )

    model_uri = str(model_uri)
    isvc_yaml = """
    apiVersion: "serving.kubeflow.org/v1beta1"
    kind: "InferenceService"
    metadata:
      name: {}
      namespace: {}
    spec:
      predictor:
        serviceAccountName: sa
        pytorch:
          storageUri: {}
          resources:
            limits:
              memory: 4Gi   
    """.format(
        DEPLOY, namespace, model_uri
    )
    deploy_task = (
        deploy_op(action="apply", inferenceservice_yaml=isvc_yaml)
        .after(minio_mar_upload)
        .set_display_name("Deployer")
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pytorch_cifar10, package_path="pytorch_cifar10.yaml")
