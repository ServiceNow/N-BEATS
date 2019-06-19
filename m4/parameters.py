training_parameters = {
    # model architecture
    'model_type': 'interpretable',  # 'generic',

    'repeat': list(range(10)),  # must always be an array even for only one repeat, for example: [1]

    # training dataset
    'training_split': 'train',  # train_subset
    'input_size': list(range(2, 8)),
    'ts_per_model_ratio': 0.2,
    'feature_bagging': 0.25,
    'batch_size': 1024,

    # training parameters
    'loss_name': ['MASE', 'MAPE', 'SMAPE'],
    'init_lr': 0.001,
    'weight_decay': 0.0,
    'iterations': 30001,

    'training_checkpoint_interval': 10000,

    # generic model parameters (these parameters will be ignored for 'interpretable' model type)
    'stacks': 30,
    'blocks_in_stack': 1,
    'block_fc_size': 512,
    'block_fc_layers': 4,

    # interpretable model parameters (these parameters will be ignored for 'generic' model type)
    'trend_blocks': 3,
    'trend_block_fc_size': 256,
    'trend_block_fc_layers': 4,
    'trend_order': 3,
    'seasonality_blocks': 3,
    'seasonality_block_fc_size': 2048,
    'seasonality_block_fc_layers': 4,
    'seasonality_num_harmonics': 1
}