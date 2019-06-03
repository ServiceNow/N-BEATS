import logging
import os
import pickle
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops, random_ops

from m4.dataset import M4Info, M4Dataset, M4DatasetSplit
from m4.settings import INPUT_MAXSIZE
from m4.utils import summary_log
from nbeats import NBeats, NBeatsStack, NBeatsBlock, NBeatsHarmonicsBlock, NBeatsPolynomialBlock


def train(training_dir_path: str,
          training_split: M4DatasetSplit,
          validation_split: Optional[M4DatasetSplit],
          training_checkpoint_interval: int,
          validation_checkpoint_interval: int,
          input_horizons: int,
          ts_per_model: int,
          features_mask,  # TODO: same as features_subset
          loss_name: str,
          batch_size: int,
          iterations: int,
          init_lr: float,
          model_fn: Callable[[int], NBeats]):
    #
    # Load data
    #
    m4_info = M4Info()

    # Load sampled time series for the training instance.
    ts_ids_file_path = os.path.join(training_dir_path, 'ts_ids.pickle')
    if not os.path.isfile(ts_ids_file_path):
        with open(ts_ids_file_path, 'wb') as f:
            pickle.dump(np.random.choice(m4_info.ids,
                                         size=int(ts_per_model * m4_info.total_number_of_timeseries),
                                         replace=False), f)

    with open(ts_ids_file_path, 'rb') as f:
        ts_ids_subset = pickle.load(f)

    training_set = M4Dataset(split=training_split)
    validation_set = M4Dataset(split=validation_split) if validation_split is not None else None

    # TODO: load features subset

    #
    # Main graph
    #
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

        targets = {}
        target_masks = {}

        # TODO: check loop inversion
        with tf.variable_scope('inputs'):
            inputs = tf.placeholder(shape=(batch_size, 1, INPUT_MAXSIZE),
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
                                                       name='targets_mask_{h}',
                                                       dtype=tf.float32)

        models = {}
        with tf.variable_scope('M4-model'):
            for horizon in m4_info.horizons:
                with tf.variable_scope(f'horizon_{horizon}', reuse=False):
                    input_size = min(input_horizons * horizon, INPUT_MAXSIZE)
                    model_input = inputs[:, input_size]

                    # delevel input of Hourly and Weekly
                    # TODO: fix UGLY mask issue
                    mask = tf.not_equal(model_input, tf.zeros_like(model_input))
                    model_input = model_input - model_input[:, :1] - 0.01
                    model_input = tf.multiply(model_input, tf.cast(mask, model_input.dtype))

                    model_input = tf.multiply(model_input, features_mask[None, :input_size], name='features_subset')
                    model = model_fn(horizon)
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
        lr_decay_step = iterations // 3  # decay 3 times
        learning_rate = tf.train.exponential_decay(init_lr, global_step, lr_decay_step, 0.5, staircase=True)
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
        log_dir_path = os.path.join(training_dir_path, 'logs')
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge(tf.get_collection('summaries'))
        summary_writer = tf.summary.FileWriter(training_dir_path, flush_secs=1)
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
            for step in range(checkpoint_step, iterations):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    batch = training_set.next_batch(batch_size=batch_size)
                    feed_dict = {
                        targets[batch.horizon]: batch.targets,
                        inputs: batch.inputs,
                        target_masks[batch.horizon]: batch.target_mask,
                        masep: batch.masep
                    }
                    batch_loss = sess.run(training_operations[batch.horizon], feed_dict=feed_dict)
                    train_log_results[f'train_loss/horizon_{batch.horizon}'] = batch_loss

                    if step % training_checkpoint_interval == 0:
                        train_log_writer(step, **train_log_results)
                        logging.info(f'step {step}, loss: {batch_loss}')
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                        saver.save(sess, os.path.join(log_dir_path, 'model'), global_step=step)
                    # TODO: validation


def generic_basis(stacks: int,
                  blocks_in_stack: int,
                  block_fc_size: int,
                  block_fc_layers: int,
                  forecast_horizon: int,
                  activation_fn,
                  weights_regularizer):
    return NBeats([NBeatsStack([NBeatsBlock(hidden_units=block_fc_size,
                                            layers=block_fc_layers,
                                            forecast_horizon=forecast_horizon,
                                            activation_fn=activation_fn,
                                            weights_regularizer=weights_regularizer
                                            )
                                for _ in range(blocks_in_stack)])
                   for _ in range(stacks)])


def interpretable_basis(trend_blocks: int,
                        trend_block_fc_size: int,
                        trend_block_fc_layers: int,
                        trend_order: int,
                        seasonality_blocks: int,
                        seasonality_block_fc_size: int,
                        seasonality_block_fc_layers: int,
                        seasonality_num_harmonics: int,
                        forecast_horizon: int,
                        activation_fn,
                        weights_regularizer):
    trend_stack = NBeatsStack([NBeatsPolynomialBlock(hidden_units=trend_block_fc_size,
                                                     layers=trend_block_fc_layers,
                                                     polynomial_order=trend_order,
                                                     forecast_horizon=forecast_horizon,
                                                     activation_fn=activation_fn,
                                                     weights_regularizer=weights_regularizer)
                               for _ in range(trend_blocks)])
    seasonality_stack = NBeatsStack([NBeatsHarmonicsBlock(hidden_units=seasonality_block_fc_size,
                                                          layers=seasonality_block_fc_layers,
                                                          num_of_harmonics=seasonality_num_harmonics,
                                                          forecast_horizon=forecast_horizon,
                                                          activation_fn=activation_fn,
                                                          weights_regularizer=weights_regularizer)
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
