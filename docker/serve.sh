#!/usr/bin/env bash
BASEDIR=$(pwd)

if [[ "$1" = '/dev/nvidia'* ]]
then
    echo "Running on GPU"
    docker run -p 43889:8080 \
        --gpus all \
        --device $1 \
        --device /dev/nvidia-uvm \
        --device /dev/nvidia-uvm-tools \
        --device /dev/nvidiactl \
        -v $BASEDIR:/home/model-server/model-store \
        -it coref-modelserver \
            torchserve \
            --ts-config /home/model-server/config.properties \
            --models droc_incremental=droc_incremental_teacherforcing_discard.mar \
            --start
else
    echo "Running on CPU"
    docker run -p 43889:8080 \
        -v $BASEDIR:/home/model-server/model-store \
        -it coref-modelserver:cpu \
            torchserve \
            --ts-config /home/model-server/config.properties \
            --models droc_incremental=droc_incremental_teacherforcing_discard.mar \
            --start
fi
