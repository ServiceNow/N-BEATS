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

init:
	docker build . -t ${IMAGE}

dataset:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} python datasets/main.py build

experiment: .require-name .require-config
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} python experiments/${name}/main.py build \
		--experiment_id=${EXPERIMENT_ID} --config=${config}

run: .require-name .require-config .require-instance
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		   bash -c "`cat ${ROOT}/storage/experiments/${EXPERIMENT_ID}/instances/${instance}/command`"

notebook: .require-port
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p $(port):8888 $(IMAGE) \
		   bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

obuild:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		python $(experiment)/main.py init --name=$(name)
orun:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		   bash -c "`cat ${ROOT}/storage/${experiment}/${instance}/command`"


.require-name:
ifndef name
	$(error name is required)
endif

.require-config:
ifndef config
	$(error config is required)
endif

.require-instance:
ifndef instance
	$(error instance is required)
endif

.require-port:
ifndef port
	$(error port is required)
endif
