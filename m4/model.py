import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops, random_ops

from m4.dataset import M4Info, M4Dataset, M4DatasetSplit
from m4.experiment import M4Experiment
from m4.settings import M4_INPUT_MAXSIZE
from m4.utils import summary_log
from nbeats import NBeats, NBeatsStack, NBeatsBlock, NBeatsHarmonicsBlock, NBeatsPolynomialBlock


def train(experiment_path: str):
    m4_info = M4Info()
    experiment = M4Experiment.load(experiment_path)

    training_set = M4Dataset(split=M4DatasetSplit[experiment.parameters.training_split.upper()])
    batch_size = experiment.parameters.batch_size
    loss_name = experiment.parameters.loss_name

    #
    # Main graph
    #
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

        targets = {}
        target_masks = {}

        # TODO: check loop inversion
        with tf.variable_scope('inputs'):
            inputs = tf.placeholder(shape=(batch_size, M4_INPUT_MAXSIZE),
                                    name='inputs',
                                    dtype=tf.float32)
            masep = tf.placeholder(shape=(batch_size,),
                                   name='masep',
                                   dtype=tf.float32)
            for horizon in m4_info.horizons:
                targets[horizon] = tf.placeholder(shape=(batch_size, horizon),
                                                  name=f'target_{horizon}',
                                                  dtype=tf.float32)
                target_masks[horizon] = tf.placeholder(shape=(batch_size, horizon),
                                                       name=f'targets_mask_{horizon}',
                                                       dtype=tf.float32)

        models = {}
        with tf.variable_scope('M4-model'):
            for horizon in m4_info.horizons:
                with tf.variable_scope(f'horizon_{horizon}', reuse=tf.AUTO_REUSE):
                    input_size = min(experiment.parameters.input_size * horizon, M4_INPUT_MAXSIZE)
                    model_input = inputs[:, :input_size]

                    # delevel input of Hourly and Weekly
                    # TODO: fix UGLY mask issue
                    # mask = tf.not_equal(model_input, tf.zeros_like(model_input))
                    # model_input = model_input - model_input[:, :1] - 0.01
                    # model_input = tf.multiply(model_input, tf.cast(mask, model_input.dtype))

                    model_input = tf.multiply(model_input, experiment.input_mask[None, :input_size],
                                              name='model_input')
                    model = interpretable_basis(input_size=input_size,
                                                trend_blocks=experiment.parameters.trend_blocks,
                                                trend_block_fc_size=experiment.parameters.trend_block_fc_size,
                                                trend_block_fc_layers=experiment.parameters.trend_block_fc_layers,
                                                trend_order=experiment.parameters.trend_order,
                                                seasonality_blocks=experiment.parameters.seasonality_blocks,
                                                seasonality_block_fc_size=experiment.parameters.seasonality_block_fc_size,
                                                seasonality_block_fc_layers=experiment.parameters.seasonality_block_fc_layers,
                                                seasonality_num_harmonics=experiment.parameters.seasonality_num_harmonics,
                                                forecast_horizon=horizon,
                                                weight_decay=experiment.parameters.weight_decay) \
                        if experiment.parameters.model_type == 'interpretable' \
                        else generic_basis(input_size=input_size,
                                           stacks=experiment.parameters.stacks,
                                           blocks_in_stack=experiment.parameters.blocks_in_stack,
                                           block_fc_size=experiment.parameters.block_fc_size,
                                           block_fc_layers=experiment.parameters.block_fc_layers,
                                           forecast_horizon=horizon,
                                           weight_decay=experiment.parameters.weight_decay)
                    models[horizon] = model.build(model_input)

        # Training operations
        training_operations = {}
        losses = {}
        for horizon in m4_info.horizons:
            with tf.variable_scope(f'loss_horizon_{horizon}', reuse=False):
                forecast = models[horizon]
                target = targets[horizon]
                # TODO: clarify masking
                target_mask = target_masks[horizon]
                if loss_name == 'MASE':
                    masked_masep_inv = tf.div_no_nan(target_mask, masep[:, None])
                    losses[horizon] = tf.reduce_mean(tf.abs(target - forecast) * masked_masep_inv)
                elif loss_name == 'MAPE':
                    weights = tf.div_no_nan(target_mask, target)
                    losses[horizon] = tf.losses.absolute_difference(labels=target,
                                                                    predictions=forecast,
                                                                    weights=weights)
                elif loss_name == 'SMAPE':
                    weights = tf.stop_gradient(tf.div_no_nan(2.0 * target_mask, (target + tf.abs(forecast))))
                    losses[horizon] = tf.reduce_mean(tf.abs(target - forecast) * weights)
                else:
                    raise Exception(f'Loss {loss_name} not implemented')

        # Learning rate
        lr_decay_step = experiment.parameters.iterations // 3  # decay 3 times
        learning_rate = tf.train.exponential_decay(experiment.parameters.init_lr, global_step, lr_decay_step, 0.5,
                                                   staircase=True)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
        for horizon in m4_info.horizons:
            total_loss = tf.add_n([losses[horizon]] + regularization_losses)
            training_operations[horizon] = slim.learning.create_train_op(total_loss=total_loss,
                                                                         optimizer=optimizer,
                                                                         global_step=global_step,
                                                                         clip_gradient_norm=1.0)

        # Training Summary
        log_dir_path = os.path.join(experiment_path, 'logs')
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge(tf.get_collection('summaries'))
        summary_writer = tf.summary.FileWriter(experiment_path, flush_secs=1)
        train_log_writer = summary_log(log_dir_path, writer=summary_writer)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        supervisor = tf.train.Supervisor(logdir=log_dir_path,
                                         init_feed_dict=None,
                                         summary_op=None,
                                         init_op=tf.global_variables_initializer(),
                                         summary_writer=None,
                                         saver=saver,
                                         global_step=global_step,
                                         save_summaries_secs=60,
                                         save_model_secs=0)
        # Main training loop
        with supervisor.managed_session() as sess:
            summary_writer.add_graph(sess.graph)
            summary_writer.flush()
            checkpoint_step = sess.run(global_step)
            if checkpoint_step > 0:
                checkpoint_step += 1
            train_log_results = dict()
            for step in range(checkpoint_step, experiment.parameters.iterations):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    batch = training_set.next_batch(batch_size=batch_size, indices_filter=experiment.timeseries_indices)
                    feed_dict = {
                        targets[batch.horizon]: batch.targets,
                        inputs: batch.inputs,
                        target_masks[batch.horizon]: batch.target_mask,
                        masep: batch.masep
                    }
                    batch_loss = sess.run(training_operations[batch.horizon], feed_dict=feed_dict)
                    train_log_results[f'train_loss/horizon_{batch.horizon}'] = batch_loss

                    if step % experiment.parameters.training_checkpoint_interval == 0:
                        train_log_writer(step, **train_log_results)
                        print(f'step {step}, loss: {batch_loss}')
                        logging.info(f'step {step}, loss: {batch_loss}')
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                        saver.save(sess, os.path.join(log_dir_path, 'model'), global_step=step)
                    # TODO: validation


def generic_basis(input_size: int,
                  stacks: int,
                  blocks_in_stack: int,
                  block_fc_size: int,
                  block_fc_layers: int,
                  forecast_horizon: int,
                  weight_decay: float,
                  activation_fn=tf.nn.relu):
    return NBeats([NBeatsStack([NBeatsBlock(input_size=input_size,
                                            hidden_units=block_fc_size,
                                            layers=block_fc_layers,
                                            forecast_horizon=forecast_horizon,
                                            activation_fn=activation_fn,
                                            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
                                for _ in range(blocks_in_stack)])
                   for _ in range(stacks)])


def interpretable_basis(input_size: int,
                        trend_blocks: int,
                        trend_block_fc_size: int,
                        trend_block_fc_layers: int,
                        trend_order: int,
                        seasonality_blocks: int,
                        seasonality_block_fc_size: int,
                        seasonality_block_fc_layers: int,
                        seasonality_num_harmonics: int,
                        forecast_horizon: int,
                        weight_decay: float,
                        activation_fn=tf.nn.relu):
    trend_stack = NBeatsStack([NBeatsPolynomialBlock(input_size=input_size,
                                                     hidden_units=trend_block_fc_size,
                                                     layers=trend_block_fc_layers,
                                                     polynomial_order=trend_order,
                                                     forecast_horizon=forecast_horizon,
                                                     activation_fn=activation_fn,
                                                     regularizer=tf.contrib.layers.l2_regularizer(
                                                         scale=weight_decay))
                               for _ in range(trend_blocks)])
    seasonality_stack = NBeatsStack([NBeatsHarmonicsBlock(input_size=input_size,
                                                          hidden_units=seasonality_block_fc_size,
                                                          layers=seasonality_block_fc_layers,
                                                          num_of_harmonics=seasonality_num_harmonics,
                                                          forecast_horizon=forecast_horizon,
                                                          activation_fn=activation_fn,
                                                          regularizer=tf.contrib.layers.l2_regularizer(
                                                              scale=weight_decay))
                                     for _ in range(seasonality_blocks)])
    return NBeats([trend_stack, seasonality_stack])


class ScaledVarianceRandomNormal(init_ops.Initializer):
    """Initializer that generates tensors with a normal distribution scaled as per https://arxiv.org/pdf/1502.01852.pdf.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, factor=1.0, seed=None, dtype=dtypes.float32):
        self.mean = mean
        self.factor = factor
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        if shape:
            n = float(shape[-1])
        else:
            n = 1.0
        for dim in shape[:-2]:
            n *= float(dim)

        self.stddev = np.sqrt(self.factor * 2.0 / n)
        return random_ops.random_normal(shape, self.mean, self.stddev,
                                        dtype, seed=self.seed)
