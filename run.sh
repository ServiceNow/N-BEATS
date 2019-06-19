#!/usr/bin/env bash

DOCKER_IMAGE_NAME=nbeats:${USER}

if [[ "$(docker images -q ${DOCKER_IMAGE_NAME} 2> /dev/null)" == "" ]]; then
  docker build . -t ${DOCKER_IMAGE_NAME}
fi

docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
--rm --user $(id -u) -v $(dirname "$(pwd)"):/project -w /project/source -e PYTHONPATH=/project/source \
${DOCKER_IMAGE_NAME} python $@