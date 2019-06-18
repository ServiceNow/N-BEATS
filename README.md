# N-BEATS Neural basis expansion analysis for interpretable time series forecasting

Implementation of [https://arxiv.org/abs/1905.10437](https://arxiv.org/abs/1905.10437)

## Instruction to reproduce M4 results

### Prerequisites

* CUDA compatible GPU (we use CUDA 10.0 by default)
* nvidia-docker 2.0


### Training and Evaluation

We use ensemble of 180 models, where models can be trained in parallel.
If your infrastructure supports clustering then we recommend to run the instructions below on a shared storage,
so that each docker container would share same code base, data source and experiments directory.
Also, make sure all the commands run where nvidia-docker 2 is available.

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
    In `m4/parameters.py` find `training_parameters` dictionary and adjust `model_type` value, if needed.
    `generic` is already set as default.
1. Initialize experiment.
    This step creates experiment directories with the settings defined in `m4_main.py` `training_parameters` dictionary.
    ```bash
    $ ./run.sh m4/main.py init_experiment
    ````
    This command will print experiment name in the end. The `../experiments/m4/<experiment-name>` directory will contain 
    model names.
1. Train a model for each experiment. On all available GPU(s):
    ```bash
    $ make experiment=<experiment-name>/<model-name> train
    ```
    Following example above the command would look like:
    ```bash
    $ make experiment=190614_174309_generic/repeat=0,input_size=2,loss_name=MAPE train
    ```

    If you want to dedicate a specific GPU for a model you can specify gpu id as following:
    ```bash
    $ make experiment=<experiment-name>/<model-name> gpus=1 train
    ```
    Note: training sessions are restartable and can be stopped and restarted any time.
    Checkpoint interval can be configured in `m4_main.py`, see `training_checkpoint_interval` key.

    Depending on your infrastructure you may want to write a script to automatically go through all models
    in the experiment directory and start training processes in parallel.
1. When all models are trained you can run summary statistics.

    Note: to make sure that training is done check if there is `predictions.csv` file in each model directory,
    containing forecast for all 100 000 time series.

    ```bash
    $ make experiment=<experiment-name> summary
    ```
    This command will build ensemble, download test set, naive2 forecast (for OWA) and calculate M4-report style metric for the ensemble.

    The result should look something like:
    ```
              Yearly  Quarterly    Monthly    Others    Average
    sMAPE  12.858686   9.365132  12.147478  3.984333  11.235136
    OWA     0.755116   0.812525   0.823808  0.876615   0.799172
    ```

#### Train and validating without test set

Change `training_split` key in `m4_main.py` to 