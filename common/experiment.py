"""
Experiment class definition.
"""
import logging
import os
import sys
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Any, Dict

import gin
from tqdm import tqdm

from common.settings import EXPERIMENTS_DIR


class Experiment(ABC):
    """
    Base class for creating and launching jobs in parallel.

    An implementation of this class must provide build and run methods as well as dictionary
    with configurations.
    """

    def __init__(self, experiment_id: str, freeze_when_done: bool = False):
        """
        Initialize experiment
        """
        self.experiment_id = experiment_id
        self.experiment_path = os.path.join(EXPERIMENTS_DIR, self.experiment_id)
        self.artifacts_path = os.path.join(self.experiment_path, 'artifacts')
        self.instances_path = os.path.join(self.experiment_path, 'instances')
        self.command_file_name = 'command'
        self.config_file_name = 'config.gin'
        self.freeze_when_done = freeze_when_done

    @classmethod
    @abstractmethod
    def experiment_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Dictionary of experiment parameters.
        For example:
            {'layers_search': {
               'repeats': [1, 2, 3],
               'layers': [2, 3, 4],
               'learning_rate': 1e-3
             },
             'test': {
               'repeats': [1],
               'layers': 1,
               'learning_rate': 1e-3
             }
            }
        """

    @abstractmethod
    def experiment_artifacts(self, config: str) -> None:
        """
        Add experiment specific artifacts which must be shared between instances.

        Note: This method is invoked after all instances are created.

        :param config: Configuration name.
        """

    def build(self, config: str) -> None:
        """
        Build experiment instances and artifacts.

        :param config: Configuration name.
        """
        if os.path.isdir(self.experiment_path):
            logging.error('\nExperiment "%s" already exists.\n'
                          'If you wish to re-build and re-run this experiment then you must '
                          'delete the experiment directory locally and remotely, '
                          'if it was deployed.\n', self.experiment_id)
            sys.exit(1)
        if config not in self.experiment_parameters():
            logging.error('\n"%s" is not registered in the experiment configuration dictionary.',
                          config)
            sys.exit(1)

        # Create experiment directories
        Path(self.experiment_path).mkdir(parents=True, exist_ok=False)
        Path(self.artifacts_path).mkdir(parents=False, exist_ok=False)
        Path(self.instances_path).mkdir(parents=False, exist_ok=False)

        experiment_configuration = self.experiment_parameters()[config]

        # generate experiment instances from parameters with array values.
        experiment_variables = []
        for key, val in experiment_configuration.items():
            if isinstance(val, list):
                experiment_variables.append([(key, element) for element in val])

        # create experiment instance(s)
        logging.info('Creating experiment instances ...')
        for variables_instance in tqdm(product(*experiment_variables)):
            instance_name = ','.join(['%s=%.4g' % (name, value) if isinstance(value, float) \
                                          else '%s=%s' % (name, str(value).replace(' ', '_')) \
                                      for name, value in dict(variables_instance).items()])
            instance_path = os.path.join(self.instances_path, instance_name)
            Path(instance_path).mkdir(parents=True, exist_ok=False)

            # write parameters json
            with open(os.path.join(instance_path, self.config_file_name), 'w') as cfg:
                for key, value in dict(**{**experiment_configuration,
                                          **dict(variables_instance)}).items():
                    if isinstance(value, str):
                        value = f"'{value}'"
                    cfg.write(f'run.{key} = {value}\n')

            # write command file
            with open(os.path.join(instance_path, self.command_file_name), 'w') as cmd:
                cmd.write(f'python {sys.modules["__main__"].__file__} load_instance '
                          f'--experiment_id={self.experiment_id} '
                          f'--instance_name={instance_name} '
                          f'>> {instance_path}/instance.log 2>&1')

        self.experiment_artifacts(config)

    def load_instance(self, instance_name: str) -> None:
        """
        If instance is not finished then load configuration, run logic,
        then mark as finished and change content to read-only.
        """
        self.instance_path = os.path.join(self.instances_path, instance_name)
        success_flag = os.path.join(self.instances_path, instance_name, '_SUCCESS')
        if os.path.isfile(success_flag):
            return
        instance_config_path = os.path.join(self.instances_path, instance_name,
                                            self.config_file_name)
        if os.path.isfile(instance_config_path):
            gin.parse_config_file(instance_config_path)
        self.run()

        # mark experiment as finished.
        Path(success_flag).touch()

        if self.freeze_when_done:
            # make experiment directory and its content read-only.
            for root, dirs, files in os.walk(os.path.join(self.instances_path, instance_name)):
                os.chmod(root, 0o555)
                for directory in dirs:
                    os.chmod(os.path.join(root, directory), 0o555)
                for file in files:
                    os.chmod(os.path.join(root, file), 0o444)
