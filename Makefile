IMAGE_NAME = nbeats
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

DOCKER_CMD = docker run -it --rm --runtime=nvidia -v ${PROJECT_DIR}:/project -w /project/source -e PYTHONPATH=/project/source ${IMAGE_NAME}

build:
	docker build . -t ${IMAGE_NAME}

load_training_dataset:
	@eval ${DOCKER_CMD} python m4_main.py load_training_dataset

init_ensembles:
	@eval ${DOCKER_CMD} python m4_main.py init_ensembles

bash:
	@eval ${DOCKER_CMD} bash

