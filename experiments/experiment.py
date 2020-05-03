import json
import logging
import os
from itertools import product
from pathlib import Path
from typing import Callable, Dict

from tqdm import tqdm

experiments_dir = os.path.join(os.sep, 'project', 'experiments')
parameters_file_name = 'parameters.json'
command_file_name = 'command'


def create_experiment(experiment_path: str,
                      parameters: Dict,
                      command: Callable[[str, Dict], str],
                      callback: Callable[[str, Dict], None] = lambda path, params: None) -> None:
    """
    Create experiment.
    If parameters contain keys with multiple values, then multiple sub-experiments will be created.
    The experiment(s) will be created in '/project/experiments/<module-name>/<timestamp>_experiment_name'.
    Each experiment directory will contain 2 files:
    - parameters.json: file which contains instance of parameters.
    - experiment.cmd: command to execute to start experiment.

    :param experiment_path: Path to experiment.
    :param parameters: Experiment parameters dictionary, for grid search specify array of values.
    :param command: Function which takes experiment full path, for container, and returns command to start the
    experiment.
    :param callback: Function which will be called for every created sub-experiment with it's full path. Default: None
    """
    # generate experiment instances from parameters with array values.
    experiment_variables = []
    for key, val in parameters.items():
        if isinstance(val, list):
            experiment_variables.append([(key, element) for element in val])

    # create experiment instance(s)
    logging.info('Generating experiments ...')
    for variables_instance in tqdm(product(*experiment_variables)):
        sub_experiment_name = ','.join(
            ['%s=%.4g' % (name, value) if isinstance(value, float) else '%s=%s' % (name, str(value).replace(' ', '_'))
             for name, value in dict(variables_instance).items()])
        sub_experiment_path = os.path.join(experiment_path, sub_experiment_name)
        Path(sub_experiment_path).mkdir(parents=True, exist_ok=False)

        # write parameters json
        with open(os.path.join(sub_experiment_path, parameters_file_name), 'w') as f:
            json.dump(dict(**{**parameters, **dict(variables_instance)}), f, indent=4)
        # write command file
        with open(os.path.join(sub_experiment_path, command_file_name), 'w') as f:
            f.write(command(sub_experiment_path, dict(variables_instance)))
        callback(sub_experiment_path, dict(**{**parameters, **dict(variables_instance)}))


def load_experiment_parameters(experiment_path: str) -> Dict:
    """
    Load experiment parameters in dictionary.

    :param experiment_path: Path to experiment directory where the parameters.json file is located.
    :return: Parameters dictionary.
    """
    with open(os.path.join(experiment_path, parameters_file_name), 'r') as f:
        return json.load(f)