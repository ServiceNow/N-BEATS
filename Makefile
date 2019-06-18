IMAGE_NAME = nbeats
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
GPUS := $(if $(gpus),$(gpus),all)

DOCKER_CMD = docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(GPUS) --rm --user $$(id -u) -v ${PROJECT_DIR}:/project -w /project/source -e PYTHONPATH=/project/source ${IMAGE_NAME}

build:
	docker build . -t ${IMAGE_NAME}

load_training_dataset:
	@eval ${DOCKER_CMD} python m4_main.py load_training_dataset

init_experiment:
	@eval ${DOCKER_CMD} python m4_main.py init_experiment

train:
	@eval ${DOCKER_CMD} python m4_main.py train --experiment $(experiment)

summary:
	@eval ${DOCKER_CMD} python m4_main.py summary --experiment $(experiment)

bash:
	@eval ${DOCKER_CMD} bash
