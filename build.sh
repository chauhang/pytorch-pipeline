#!/bin/bash

## Generating current timestamp
python3 gen_image_timestamp.py > curr_time.txt

export images_tag=$(cat curr_time.txt)
echo ++++ Building component images with tag=$images_tag


full_image_name=jagadeeshj/pytorch_pipeline:$images_tag

echo IMAGE TO BUILD: $full_image_name

export full_image_name=$full_image_name


## build and push docker - to fetch the latest changes and install dependencies
cd pytorch_pipeline

docker build --no-cache -t $full_image_name .
docker push $full_image_name

cd ..

## Update component.yaml files with the latest docker image name

find $1 -name "*.yaml" | grep -v 'deploy' | grep -v "tensorboard"  | while read -d $'\n' file
do
    yq -i eval ".implementation.container.image =  \"$full_image_name\"" $file
done


## compile pipeline

echo Running pipeline compilation
echo "$1/pipeline.py"
python3 "$1/pipeline.py" --target kfp

