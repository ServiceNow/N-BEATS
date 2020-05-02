include Makefile

IMAGE := images.borgy.elementai.net/nbeats:${USER}

push:
	docker push ${IMAGE}

upload-code:
	rsync -az --no-motd \
	 	  --exclude '.git' \
	 	  --exclude '.idea' \
	 	  --exclude 'storage' \
	 	  * dmitri@dc1-wks-01.elementai.net:/mnt/scratch/dmitri/nbeats

run-all:
	for instance in $$(ls ${ROOT}/storage/experiments/${EXPERIMENT_ID}/instances); do \
		borgy submit \
				--image=${IMAGE} \
				-v ${ROOT}:/experiment \
				-w /experiment \
				-e PYTHONPATH=/experiment \
				-e STORAGE=/experiment/storage \
				--cpu=1 --gpu=1 --mem=64 --bid=0 --restartable -- \
				bash -c "`cat ${ROOT}/storage/experiments/${EXPERIMENT_ID}/instances/$${instance}/command`"; \
	done