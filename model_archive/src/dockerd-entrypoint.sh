#!/bin/bash
########)#############################################################
#### Script for model archiving and config.properties generation #####
######################################################################
set -e

while [ $# -gt 0 ]; 
do 
    case $1 in
        -o|--output_path) OUTPUT_PATH="$2"; shift;shift;;
        -i|--input_path) INPUT_PATH="$2"; shift;shift;;
        -p|--properties) PROPERTIES="$2"; shift;shift;;
        -s|--serializedfile) SERIALIZEDFILE="$2"; shift;shift;;
        -h|--handlerfile) HANDLERFILE="$2"; shift;shift;;
        -m|--modelfile) MODELFILE="$2"; shift;shift;;
        -e|--extrafiles) EXTRAFILES="$2"; shift;shift;;
        -r|--requirements) REQUIREMENTS="$2"; shift;shift;;
    *) echo "! Unknown parameter passed: $1"; shift;shift;;
    esac; 
done

EXPORT_PATH=$OUTPUT_PATH
mkdir -p "$EXPORT_PATH"

download_file () {
    if [[ $1 =~ ^(http|https)://.* ]]; then
        wget "$1" -P "$2" --no-check-certificate
    fi
}

FILES=("$PROPERTIES" "$SERIALIZEDFILE" "$HANDLERFILE" "$MODELFILE" "$EXTRAFILES" "$REQUIREMENTS")

echo "${INPUT_PATH}/properties.json"
if [[ -f "${INPUT_PATH}/properties.json" ]];
then
    cp -r "${INPUT_PATH}/." "$EXPORT_PATH"
else
    for FILE in "${FILES[@]}";
    do
        download_file "$FILE" "$EXPORT_PATH"
    done
fi

MODEL_STORE="${EXPORT_PATH}/model-store/"
CONFIG_PATH="${EXPORT_PATH}/config/"

mkdir -p "$CONFIG_PATH"
mkdir -p "$MODEL_STORE"
touch "${CONFIG_PATH}/config.properties"


cat <<EOF > "${CONFIG_PATH}/config.properties"
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
number_of_netty_threads=4
job_queue_size=100
service_envelope=kfserving
install_py_dep_per_model=true
model_store=/mnt/models/model-store
model_snapshot=
EOF

CONFIG_PROPERTIES="${CONFIG_PATH}/config.properties"
truncate -s -1 "$CONFIG_PROPERTIES"

PROPERTIES_JSON="${EXPORT_PATH}/properties.json"

count=$(jq -c '. | length' "$PROPERTIES_JSON")
echo "{\"name\":\"startup.cfg\",\"modelCount\":\"3\",\"models\":{}}" | jq -c --arg count "${count}" '.["modelCount"]=$count' >> $CONFIG_PROPERTIES
sed -i 's/{}}//' "$CONFIG_PROPERTIES"
truncate -s -1 "$CONFIG_PROPERTIES"
# shellcheck disable=SC1091
jq -c '.[]' "$PROPERTIES_JSON" | while read -r i; do
    modelName=$(echo "$i" | jq -r '."model-name"')
    modelFile=$(echo "$i" | jq -r '."model-file"')
    version=$(echo "$i" | jq -r '."version"')
    serializedFile=$(echo "$i" | jq -r '."serialized-file"')
    extraFiles=$(echo "$i" | jq -r '."extra-files"')
    handler=$(echo "$i" | jq -r '."handler"')
    minWorkers=$(echo "$i" | jq -r '."min-workers"')
    maxWorkers=$(echo "$i" | jq -r '."max-workers"')
    batchSize=$(echo "$i" | jq -r '."batch-size"')
    maxBatchDelay=$(echo "$i" | jq -r '."max-batch-delay"')
    responseTimeout=$(echo "$i" | jq -r '."response-timeout"')
    marName=${modelName}.mar
    requirements=$(echo "$i" | jq -r '."requirements"')
    updatedExtraFiles=$(echo "$extraFiles" | tr "," "\n" | awk -v base_path="$EXPORT_PATH" '{ print base_path"/"$1 }' | paste -sd "," -)
    ######################################
    #### Support for custom handlers #####
    ######################################
    pyfile="$( cut -d '.' -f 2 <<< "$handler" )"
    if [ "$pyfile" == "py" ];
    then
        handler="${EXPORT_PATH}/${handler}"
    fi
    ## 
    if [ -z "${requirements}" ];    # If requirements is empty string or unset
    then
        touch "${EXPORT_PATH}/requirements.txt"
    fi
    if [ -n "${modelFile}" ];   # If modelFile is non empty string
    then
        if [ -n "${extraFiles}" ];  # If extraFiles is non empty string
        then
            torch-model-archiver --model-name "$modelName" --version "$version" --model-file "${EXPORT_PATH}/${modelFile}" --serialized-file "${EXPORT_PATH}/${serializedFile}" --export-path "$MODEL_STORE" --extra-files "$updatedExtraFiles" --handler "$handler" -r "${EXPORT_PATH}/requirements.txt" --force
        else
            torch-model-archiver --model-name "$modelName" --version "$version" --model-file "${EXPORT_PATH}/${modelFile}" --serialized-file "${EXPORT_PATH}/${serializedFile}" --export-path "$MODEL_STORE" --handler "$handler" -r "${EXPORT_PATH}/requirements.txt" --force
        fi
    else
        if [ -n "${extraFiles}" ];  # If extraFiles is non empty string
        then
            torch-model-archiver --model-name "$modelName" --version "$version" --serialized-file "${EXPORT_PATH}/${serializedFile}" --export-path "$MODEL_STORE" --extra-files "$updatedExtraFiles" --handler "$handler" -r "${EXPORT_PATH}/requirements.txt" --force
        else
            torch-model-archiver --model-name "$modelName" --version "$version" --serialized-file "${EXPORT_PATH}/${serializedFile}" --export-path "$MODEL_STORE" --handler "$handler" -r "${EXPORT_PATH}/requirements.txt" --force
        fi
    fi
    echo "{\"modelName\":{\"version\":{\"defaultVersion\":true,\"marName\":\"sample.mar\",\"minWorkers\":\"sampleminWorkers\",\"maxWorkers\":\"samplemaxWorkers\",\"batchSize\":\"samplebatchSize\",\"maxBatchDelay\":\"samplemaxBatchDelay\",\"responseTimeout\":\"sampleresponseTimeout\"}}}" | 
    jq -c --arg modelName "$modelName" --arg version "$version" --arg marName "$marName" --arg minWorkers "$minWorkers" --arg maxWorkers "$maxWorkers" --arg batchSize "$batchSize" --arg maxBatchDelay "$maxBatchDelay" --arg responseTimeout "$responseTimeout" '.[$modelName]=."modelName" | .[$modelName][$version]=.[$modelName]."version" | .[$modelName][$version]."marName"=$marName | .[$modelName][$version]."minWorkers"=$minWorkers | .[$modelName][$version]."maxWorkers"=$maxWorkers | .[$modelName][$version]."batchSize"=$batchSize | .[$modelName][$version]."maxBatchDelay"=$maxBatchDelay | .[$modelName][$version]."responseTimeout"=$responseTimeout | del(."modelName", .[$modelName]."version")'  >> "$CONFIG_PROPERTIES"
    truncate -s -1 "$CONFIG_PROPERTIES"
done
sed -i 's/}{/,/g' "$CONFIG_PROPERTIES"
sed -i 's/}}}/}}}}/g' "$CONFIG_PROPERTIES"

# prevent docker exit
# tail -f /dev/null


