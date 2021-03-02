import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["resnet","bert"], required=True, help='Target model for compilation')
parser.add_argument('--target', type=str, choices=["kfp","mp"], required=True, help='Target platform for compilation')

args = parser.parse_args()

is_kfp = args.target == "kfp"

if is_kfp:
    print("Building for KFP backend")
else:
    print("Building for Managed Pipelines backend")

#import kfp.gcp as gcp
import kfp

if is_kfp:
    from kfp.components import load_component_from_file, load_component_from_url
    from kfp import dsl
    from kfp import compiler
else:
    from kfp.v2.components import load_component_from_file, load_component_from_url
    from kfp.v2 import dsl
    from kfp.v2 import compiler

data_prep_op = load_component_from_file(f"data_prep_step/{args.model}/component.yaml")
train_model_op = load_component_from_file(f"training_step/{args.model}/component.yaml")

model_archive_op = load_component_from_file("model_archive_step/component.yaml")
# deploy_model_op = load_component_from_file("kfserving/component.yaml")

# globals
USER='pavel'
PIPELINE_ROOT = 'gs://managed-pipeline-test-bugbash/20210130/pipeline_root/{}'.format(USER)

@dsl.pipeline(
    name = "pytorchcnn",
    output_directory = PIPELINE_ROOT
)
def train_imagenet_cnn_pytorch(
    ):
        
    # data_prep_task = data_prep_op(input_data = "gs://cloud-ml-nas-public/classification/imagenet/train*")
    # data_prep_task = data_prep_op(input_data = "gs://cloud-ml-nas-public/classification/imagenet/train*", vocab_file = "bert_base_uncased_vocab.txt", vocab_file_url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt")

    #temp_input = "gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2878573190/output_data/prefix"
    #data_prep_task.outputs["output_data"])

    # train_model_task = (train_model_op(trainingdata = data_prep_task.outputs["output_data"]).
    #     set_cpu_limit('4').
    #     set_memory_limit('14Gi')
    # )
    
    data_prep_task = data_prep_op(input_data = "gs://cloud-ml-nas-public/classification/imagenet/train*", vocab_file = "bert_base_uncased_vocab.txt", vocab_file_url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt")

    train_model_task = (train_model_op(trainingdata = data_prep_task.outputs["output_data"], maxepochs = 2, 
        numsamples = 150, 
        batchsize = 16,
        numworkers = 2,
        learningrate = 0.001,
        accelerator = "")
        .set_cpu_limit('4').
        set_memory_limit('14Gi')
    )


    model_archive_task = model_archive_op(model_directory = "/tmp/models/")

    # deploy_model_task = deploy_model_op(
	# 	action = 'create',
	# 	model_name='pytorch',
	# 	default_model_uri='gs://kfserving-samples/models/pytorch/cifar10/',
	# 	namespace='admin',
	# 	framework='pytorch',
	# 	default_custom_model_spec='{}',
	# 	canary_custom_model_spec='{}',
	# 	autoscaling_target='0',
	# 	kfserving_endpoint=''
	# )


if is_kfp:
    compiler.Compiler().compile(
        pipeline_func = train_imagenet_cnn_pytorch,
        #pipeline_root = PIPELINE_ROOT, this doesn't work for some reason
        package_path="pytorch_dpa_demo_kfp.yaml",
    )
else:
    compiler.Compiler().compile(
        pipeline_func = train_imagenet_cnn_pytorch,
        pipeline_root = PIPELINE_ROOT,
        output_path="pytorch_dpa_demo.json",
    )

    
'''
Namespace(
    checkpoint_root='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2069872589/ModelCheckpoint', 
    tensorboard_root='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2069872589/TensorboardLogs', 
    train_glob='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2878573190/output_data')
'''