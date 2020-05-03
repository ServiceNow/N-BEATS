"""
M3 Experiment Parameters
"""

COMMON = {
    'repeat': list(range(10)),
    'validation_mode': False,

    # training dataset
    'input_size': list(range(2, 8)),
    'training_batch_size': 1024,

    # training parameters
    'loss_name': ['MASE', 'MAPE', 'SMAPE'],
    'learning_rate': 0.001,
    'weight_decay': 0.0,
    'snapshot_frequency': 1000,
    'logging_frequency': 100,

    # architecture
    'fc_layers': 4,

    # interpretable
    'seasonality_fc_layers_size': 2048,
    'seasonality_blocks': 3,
    'num_of_harmonics': 1,

    'trend_fc_layers_size': 256,
    'degree_of_polynomial': 2,
    'trend_blocks': 3,

    # generic
    'generic_fc_layers_size': 512,
    'generic_blocks': 30,
}

INTERPRETABLE = {
    'model_type': 'interpretable',
    'history_size': {
        'M3Year': 20,
        'M3Quart': 5,
        'M3Month': 5,
        'M3Other': 20
    },
    'iterations': {
        'M3Year': 50,
        'M3Quart': 6000,
        'M3Month': 6000,
        'M3Other': 250
    }
}

GENERIC = {
    'model_type': 'generic',
    'history_size': {
        'M3Year': 20,
        'M3Quart': 20,
        'M3Month': 20,
        'M3Other': 10
    },
    'iterations': {
        'M3Year': 20,
        'M3Quart': 250,
        'M3Month': 10000,
        'M3Other': 250
    }
}

VERIFY = {
    'repeat': list(range(3))
}

PARAMETERS = {
    'interpretable': {**COMMON, **INTERPRETABLE},
    'generic': {**COMMON, **GENERIC},
    'interpretable_v': {**COMMON, **INTERPRETABLE, **VERIFY},
    'generic_v': {**COMMON, **GENERIC, **VERIFY}
}