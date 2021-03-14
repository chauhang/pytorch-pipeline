# MAR File creation component

This component creates MAR file for torchserve.

## Loading from local path

1. Place all the required files mentioned in the properties.json and the properties.json in a local path.
2. Load the component in pipeline from file
```
model_archive_op = load_component_from_file("model_archive_step/component.yaml")
```
3. use the component with `input_directory` argument as shown below. Here the output of task op hold the files for MAR creation. 
```
    model_archive_task = model_archive_op(
      input_directory=task.outputs['output'],
      handlerfile="image_classifier")
```
3. The config.properties and MAR file will be generated in the `/tmp/outputs/output_directory/data` directory after successful run.

## Loading from URLs

1. Load the component in pipeline from file
```
model_archive_op = load_component_from_file("model_archive_step/component.yaml")
```
2. use the component with all the argument except `input_directory` as shown below
```    
  model_archive_task = model_archive_op(
    properties="https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/properties.json",
    modelfile="https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/model.py", 
    serializedfile="https://download.pytorch.org/models/resnet18-5c106cde.pth", 
    extrafiles="https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/index_to_name.json", 
    handlerfile="image_classifier")
```
3. The config.properties and MAR file will be generated in the `/tmp/outputs/output_directory/data` directory after successful run.

The expected folder structure along with files is created as below.

```bash
├── config
│   ├── config.properties
├── model-store
│   ├── mnist.mar
```