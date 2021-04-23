#!/usr/bin/env sh
ARCHIVER=torch-model-archiver

show_usage() {
    echo "Usage $0 MODEL_NAME WEIGHTS [VERSION]"
    echo ""
    echo "Build model-archive using the config MODEL_NAME with weights from the WEIGHTS file"
    echo ""
}


if [ ! -z $3 ]
then
    VERSION=$3
else
    VERSION="0.0.1"
fi


if [ $# -lt 2 ]
then
    show_usage
else
    $ARCHIVER \
        --model-name $1 \
        --model-file torch_serve/incremental.py \
        --serialized-file $2 \
        --extra-files $(ls -1p | grep -v / | xargs echo | sed 's/ /,/g'),experiments.conf \
        --handler torch_serve/model_handler.py \
        --archive-format no-archive \
        --requirements-file requirements.txt \
        -v $VERSION \
        --runtime python3 \
        -f
fi
