default_parameters = {
    'repeat': list(range(10)),

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

    # generic
    'generic_blocks': 30,
    'generic_fc_layers_size': 512,

    # interpretable
    'trend_blocks': 3,
    'trend_fc_layers_size': 256,
    'degree_of_polynomial': 3,

    'seasonality_blocks': 3,
    'seasonality_fc_layers_size': 2048,
    'num_of_harmonics': 1
}


test = {**default_parameters,
        'validation_mode': False,
        'repeat': list(range(10))
        }

parameters = {
    'test': {**test,
             'repeat': [0],
             'model_type': 'interpretable',
             'iterations': 20,
             'history_size': 1
             },
    'best_generic': {**test,
                     'model_type': 'generic',
                     'history_size': {
                         'M3Year': 20,
                         'M3Quart': 250,
                         'M3Month': 10000,
                         'M3Other': 250
                     },
                     'iterations': {
                         'M3Year': 20,
                         'M3Quart': 250,
                         'M3Month': 10000,
                         'M3Other': 250
                     }
                     },
    'best_interpretable': {**test,
                           'model_type': 'interpretable',
                           'history_size': {
                               'M3Year': 20,
                               'M3Quart': 250,
                               'M3Month': 10000,
                               'M3Other': 250
                           },
                           'iterations': {
                               'M3Year': 20,
                               'M3Quart': 10000,
                               'M3Month': 10000,
                               'M3Other': 650
                           }
                           }
}
