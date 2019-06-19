# N-BEATS Neural basis expansion analysis for interpretable time series forecasting

Implementation of [https://arxiv.org/abs/1905.10437](https://arxiv.org/abs/1905.10437)

## Instruction to reproduce M4 results

### Prerequisites

* CUDA compatible GPU (we use CUDA 10.0 by default)
* nvidia-docker 2.0


### Training and Evaluation

We use an ensemble of 180 models, where models can be trained in parallel.
If you have multiple GPUs then we recommend to run the instructions below on a shared storage,
so that each docker container (1 per GPU) would share same code base, data source and experiments directory.

Make sure all the commands run where `nvidia-docker 2` is available.

Note: the training time of one model is about 50-60 minutes on NVidia Tesla v100 16GB GPU with ~50% load.

1. Create a directory and clone the project to `source` subdirectory.
    ```bash
    $ mkdir -p <project-path>
    $ cd <project-path>
    $ git clone git@github.com:ElementAI/N-BEATS.git source
    ```
1. Download training set.
    ```bash
    $ cd source
    $ ./run.sh m4/main.py download_training_dataset
    ```
    This command will build docker image, download, unpack and cache (in npz format) M4 training dataset.
    The files will be stored in `<project-path>/dataset/m4`.
1. Select type of the model to train: `generic` or `interpretable`.
    In `m4/parameters.py` find and adjust `model_type` value, if needed.
    `interpretable` is already set as default.
1. Initialize experiment.
    This step creates experiment directories with the settings defined in `m4/parameters.py`.
    ```bash
    $ ./run.sh m4/main.py init_experiment
    ```
    This command will print experiment name in the end. The `../experiments/m4/<experiment-name>` directory will contain 
    model names.
1. Train each model
    ```bash
    $ NVIDIA_VISIBLE_DEVICES=<gpu-id> ./run.sh m4/main.py train --experiment=<experiment-name> --model=<model-name>
    ```
    
    `NVIDIA_VISIBLE_DEVICES` is optional, if not provided the model will be trained on all available GPUs. However,
    if you have multiple GPUs we recommend to run one model per GPU in parallel.

1. When all models are trained run the summary statistics.

    Before proceeding to this step make sure that training is done for all models: 
    check if there is `predictions.csv` file in each model directory, containing forecast for all 100000 time series.

    ```bash
    $ ./run.sh m4/main.py summary --experiment=<experiment-name>
    ```
    This command will build ensemble, download test set, naive2 forecast (for OWA) and calculate M4-report style metric 
    for the ensemble.

    Output:
    ```
              Yearly  Quarterly    Monthly    Others    Average
    sMAPE  12.858686   9.365132  12.147478  3.984333  11.235136
    OWA     0.755116   0.812525   0.823808  0.876615   0.799172
    ```

#### Train and validate on training set only

1. Change `training_split` key in `m4/parameters.py` to `train_subset` then initialize experiment and train all models.
1. For summary step, run: `$ ./run.sh m4/main.py summary --experiment=<experiment-name> --validation=True`

Note: The summary for validation set contains SMAPE only, no OWA.