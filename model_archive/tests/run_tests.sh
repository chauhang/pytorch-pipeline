#!/bin/bash

# Files in Local Path
# ../src/dockerd-entrypoint.sh --output_path ./output --input_path ./input

# Files in Remote Path
# ../src/dockerd-entrypoint.sh --output_path ./output --input_path "" --properties "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/properties.json" \
#   --serializedfile "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
#   --handlerfile "image_classifier" \
#   --modelfile "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/model.py" \
#   --extrafiles "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/model_archive/index_to_name.json" \
#   --requirements ""
