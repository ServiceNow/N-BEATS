IMAGE_NAME = nbeats
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

DOCKER_CMD = docker run --rm -v ${PROJECT_DIR}:/project -w /project -e PYTHONPATH=/project/source ${IMAGE_NAME}

build:
	docker build . -t ${IMAGE_NAME}

load_training_dataset:
	@eval ${DOCKER_CMD} bash -c 'python m4_main.py load_training_dataset'

