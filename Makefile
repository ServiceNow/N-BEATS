IMAGE_NAME=nbeats

build:
	docker build . -t ${IMAGE_NAME}

laod_training_dataset:
	python m4_main.py load_training_dataset
