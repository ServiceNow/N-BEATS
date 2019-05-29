IMAGE := nbeats
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))

EXPERIMENT_ID = $(subst /,_,${name})_${config}

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/experiment \
	-w /experiment \
	-e PYTHONPATH=/experiment \
	-e STORAGE=/experiment/storage

ifdef gpu
	DOCKER_PARAMETERS += --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(gpu)
endif

.PHONY: test

init:
	docker build . -t ${IMAGE}

dataset:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} python datasets/main.py build

build: .require-config
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
 		python $(dir ${config})main.py --config_path=${config} build_ensemble

run: .require-command
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		   bash -c "`cat ${ROOT}/${command}`"

notebook: .require-port
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p $(port):8888 $(IMAGE) \
		   bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

test:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} python -m unittest

.require-config:
ifndef config
	$(error config is required)
endif

.require-command:
ifndef command
	$(error command is required)
endif

.require-port:
ifndef port
	$(error port is required)
endif
