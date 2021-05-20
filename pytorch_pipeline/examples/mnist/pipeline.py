import kfp
import json
from kfp.onprem import use_k8s_secret
from kfp import components
from kfp.components import load_component_from_file
from kfp import dsl
from kfp import compiler


yaml_folder_path = "examples/mnist/yaml"
log_bucket = "mlpipeline"
ax_path = f"ax/summary/{dsl.RUN_ID_PLACEHOLDER}"

prepare_tensorboard_op = load_component_from_file(f"{yaml_folder_path}/tensorboard/component.yaml")
train_op = components.load_component_from_file(f"{yaml_folder_path}/train/component.yaml")
deploy_op = load_component_from_file(f"{yaml_folder_path}/deploy/component.yaml")
pred_op = components.load_component_from_file(f"{yaml_folder_path}/prediction/component.yaml")


minio_op = components.load_component_from_file(f"{yaml_folder_path}/minio/component.yaml")


@dsl.pipeline(name="Training Cifar10 pipeline", description="Cifar 10 dataset pipeline")
def pytorch_mnist(
    minio_endpoint="http://minio-service.kubeflow:9000",
    log_dir=f"tensorboard/logs/{dsl.RUN_ID_PLACEHOLDER}/",
    mar_path=f"mar/{dsl.RUN_ID_PLACEHOLDER}/model-store",
    config_prop_path=f"mar/{dsl.RUN_ID_PLACEHOLDER}/config",
    model_uri=f"s3://mlpipeline/mar/{dsl.RUN_ID_PLACEHOLDER}",
    tf_image="jagadeeshj/tb_plugin:v1.8",
    log_bucket="mlpipeline",
    input_req="https://kubeflow-dataset.s3.us-east-2.amazonaws.com/mnist_ax/input.json",
    cookie="cookie",
    ingress_gateway="http://istio-ingressgateway.istio-system.svc.cluster.local",
    isvc_name="torchserve.kubeflow-user-example-com.example.com",
    deploy="torchserve",
    model="mnist",
    namespace="kubeflow-user-example-com",
    ax_path=f"ax/summary/{dsl.RUN_ID_PLACEHOLDER}",
):
    pod_template_spec = json.dumps(
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
                            {"name": "S3_ENDPOINT", "value": f"{minio_endpoint}"},
                            {"name": "S3_USE_HTTPS", "value": "0"},
                            {"name": "S3_VERIFY_SSL", "value": "0"},
                        ]
                    }
                ]
            }
        }
    )

    prepare_tb_task = prepare_tensorboard_op(
        log_dir_uri=f"s3://{log_bucket}/{log_dir}",
        image=tf_image,
        pod_template_spec=pod_template_spec,
    ).set_display_name("Visualization")

    train_task = (
        train_op(
            summary_url=f"minio://{log_bucket}/{ax_path}",
        )
        .apply(
            use_k8s_secret(
                secret_name="mlpipeline-minio-artifact",
                k8s_secret_key_to_env={
                    "secretkey": "MINIO_SECRET_KEY",
                    "accesskey": "MINIO_ACCESS_KEY",
                },
            )
        )
        .set_display_name("Training")
    )
    
    minio_tb_upload = (
        minio_op(
            bucket_name="mlpipeline",
            folder_name=log_dir,
            input_path=train_task.outputs["tensorboard_root"],
            filename="",
        )
        .apply(
            use_k8s_secret(
                secret_name="mlpipeline-minio-artifact",
                k8s_secret_key_to_env={
                    "secretkey": "MINIO_SECRET_KEY",
                    "accesskey": "MINIO_ACCESS_KEY",
                },
            )
        )
        .after(train_task)
        .set_display_name("Tensorboard Events Pusher")
    )
    minio_mar_upload = (
        minio_op(
            bucket_name="mlpipeline",
            folder_name=mar_path,
            input_path=train_task.outputs["checkpoint_dir"],
            filename="mnist_best.mar",
        )
        .apply(
            use_k8s_secret(
                secret_name="mlpipeline-minio-artifact",
                k8s_secret_key_to_env={
                    "secretkey": "MINIO_SECRET_KEY",
                    "accesskey": "MINIO_ACCESS_KEY",
                },
            )
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
        .apply(
            use_k8s_secret(
                secret_name="mlpipeline-minio-artifact",
                k8s_secret_key_to_env={
                    "secretkey": "MINIO_SECRET_KEY",
                    "accesskey": "MINIO_ACCESS_KEY",
                },
            )
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
        deploy, namespace, model_uri
    )
    deploy_task = (
        deploy_op(action="apply", inferenceservice_yaml=isvc_yaml)
        .after(minio_mar_upload)
        .set_display_name("Deployer")
    )
    pred_task = (
        pred_op(
            host_name=isvc_name,
            input_request=input_req,
            cookie=f"authservice_session={cookie}",
            url=ingress_gateway,
            model=model,
            inference_type="predict",
        )
        .after(deploy_task)
        .set_display_name("Prediction")
    )
    explain_task = (
        pred_op(
            host_name=isvc_name,
            input_request=input_req,
            cookie=f"authservice_session={cookie}",
            url=ingress_gateway,
            model=model,
            inference_type="explain",
        )
        .after(pred_task)
        .set_display_name("Explanation")
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pytorch_mnist, package_path="pytorch_mnist_ax.yaml")
