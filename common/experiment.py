"""
Experiment class
"""
import logging
import os
import sys
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from shutil import copy
from typing import List

import gin
from tqdm import tqdm

from common.settings import EXPERIMENTS_PATH


class Experiment(ABC):
    """
    Experiment base class.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.root = Path(config_path).parent
        self.freeze_when_done = False

        gin.parse_config_file(self.config_path)

    @abstractmethod
    def instance(self):
        """"
        Instance logic method must be implemented with @gin.configurable()
        """

    def build_ensemble(self):
        """
        Build ensemble from the given configuration.
        :return:
        """
        if EXPERIMENTS_PATH in str(self.root):
            raise Exception('Cannot build ensemble from ensemble member configuration.')
        self.build()

    @gin.configurable()
    def build(self,
              experiment_name: str,
              repeats: int,
              lookbacks: List[int],
              losses: List[str]):
        # create experiment instance(s)
        logging.info('Creating experiment instances ...')
        experiment_path = os.path.join(EXPERIMENTS_PATH, experiment_name)
        ensemble_variables = [list(range(repeats)), lookbacks, losses]
        variable_names = ['repeat', 'lookback', 'loss']
        for instance_values in tqdm(product(*ensemble_variables)):
            instance_variables = dict(zip(variable_names, instance_values))
            instance_name = ','.join(['%s=%.4g' % (name, value) if isinstance(value, float) \
                                          else '%s=%s' % (name, str(value).replace(' ', '_')) \
                                      for name, value in instance_variables.items()])
            instance_path = os.path.join(experiment_path, instance_name)
            Path(instance_path).mkdir(parents=True, exist_ok=False)

            # write parameters
            instance_config_path = os.path.join(instance_path, 'config.gin')
            copy(self.config_path, instance_config_path)
            with open(instance_config_path, 'a') as cfg:
                for name, value in instance_variables.items():
                    value = f"'{value}'" if isinstance(value, str) else str(value)
                    cfg.write(f'instance.{name} = {value}\n')

            # write command file
            command_file = os.path.join(instance_path, 'command')
            with open(command_file, 'w') as cmd:
                cmd.write(f'python {sys.modules["__main__"].__file__} '
                          f'--config_path={instance_config_path} '
                          f'run >> {instance_path}/instance.log 2>&1')

    def run(self):
        """
        Run instance logic.
        """
        success_flag = os.path.join(self.root, '_SUCCESS')
        if os.path.isfile(success_flag):
            return

        self.instance()

        # mark experiment as finished.
        Path(success_flag).touch()

        if self.freeze_when_done:
            # make experiment directory and its content read-only.
            for root, dirs, files in os.walk(self.root):
                os.chmod(root, 0o555)
                for directory in dirs:
                    os.chmod(os.path.join(root, directory), 0o555)
                for file in files:
                    os.chmod(os.path.join(root, file), 0o444)
