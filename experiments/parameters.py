from typing import Dict, NamedTuple, Union

from experiments.experiment import load_experiment_parameters


class Parameters(NamedTuple):
    repeat: int

    # training dataset
    validation_mode: bool
    input_size: int
    history_size: Union[int, Dict[str, int]]

    # training parameters
    loss_name: str
    learning_rate: float
    weight_decay: float
    iterations: Union[int, Dict[str, int]]
    training_batch_size: int
    snapshot_frequency: int
    logging_frequency: int

    # architecture
    model_type: str
    fc_layers: int

    # generic
    generic_blocks: int
    generic_fc_layers_size: int

    # interpretable
    trend_blocks: int
    trend_fc_layers_size: int
    degree_of_polynomial: int

    seasonality_blocks: int
    seasonality_fc_layers_size: int
    num_of_harmonics: int

    def history_size_for(self, group):
        return self.history_size[group] if type(self.history_size) == dict else self.history_size

    def iterations_for(self, group):
        return self.iterations[group] if type(self.iterations) == dict else self.iterations

    def snapshot_frequency_for(self, group):
        return min(self.snapshot_frequency, self.iterations_for(group))

    def logging_frequency_for(self, group):
        return min(self.logging_frequency, self.iterations_for(group) // 3)

    @staticmethod
    def load(path: str) -> 'Parameters':
        return Parameters(**load_experiment_parameters(path))