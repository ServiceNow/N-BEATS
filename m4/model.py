import os
from typing import Dict, Iterable

import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim

from m4.dataset import M4Dataset, M4DatasetSplit
from m4.experiment import M4Experiment
from m4.settings import M4_INPUT_MAXSIZE, M4_PREDICTION_FILE_NAME
from m4.utils import summary_log
from nbeats import NBeats, NBeatsStack, NBeatsBlock, SeasonalityBlock, TrendBlock


def model_graph(input_placeholder, input_mask_placeholder, experiment: M4Experiment, horizons: Iterable) -> Dict:
    models = {}
    with tf.variable_scope('M4-model', reuse=tf.AUTO_REUSE):
        for horizon in horizons:
            with tf.variable_scope(f'horizon_{horizon}', reuse=tf.AUTO_REUSE):
                input_size = min(experiment.parameters.input_size * horizon, M4_INPUT_MAXSIZE)
                model_input = input_placeholder[:, :input_size]
                input_mask = tf.multiply(input_mask_placeholder[:, :input_size],
                                         experiment.input_dropout[None, :input_size])
                apply_input_delevel = horizon == 13 or horizon == 48

                # delevel input of Hourly and Weekly
                input_level = model_input[:, :1]
                if apply_input_delevel:
                    model_input = model_input - input_level

                if experiment.parameters.model_type == 'generic':
                    model = NBeats([NBeatsStack([NBeatsBlock(input_size=input_size,
                                                             hidden_units=experiment.parameters.block_fc_size,
                                                             layers=experiment.parameters.block_fc_layers,
                                                             forecast_horizon=horizon,
                                                             activation_fn=tf.nn.relu,
                                                             regularizer=tf.contrib.layers.l2_regularizer(
                                                                 scale=experiment.parameters.weight_decay))
                                                 for _ in range(experiment.parameters.blocks_in_stack)])
                                    for _ in range(experiment.parameters.stacks)])
                elif experiment.parameters.model_type == 'interpretable':
                    trend_stack = NBeatsStack([TrendBlock(input_size=input_size,
                                                          hidden_units=experiment.parameters.trend_block_fc_size,
                                                          layers=experiment.parameters.trend_block_fc_layers,
                                                          polynomial_order=experiment.parameters.trend_order,
                                                          forecast_horizon=horizon,
                                                          activation_fn=tf.nn.relu,
                                                          regularizer=tf.contrib.layers.l2_regularizer(
                                                                         scale=experiment.parameters.weight_decay))
                                               for _ in range(experiment.parameters.trend_blocks)])
                    seasonality_stack = NBeatsStack([SeasonalityBlock(input_size=input_size,
                                                                      hidden_units=experiment.parameters.seasonality_block_fc_size,
                                                                      layers=experiment.parameters.seasonality_block_fc_layers,
                                                                      num_of_harmonics=experiment.parameters.seasonality_num_harmonics,
                                                                      forecast_horizon=horizon,
                                                                      activation_fn=tf.nn.relu,
                                                                      regularizer=tf.contrib.layers.l2_regularizer(
                                                                              scale=experiment.parameters.weight_decay))
                                                     for _ in range(experiment.parameters.seasonality_blocks)])
                    model = NBeats([trend_stack, seasonality_stack])
                else:
                    raise Exception(f'Unknown model type {experiment.parameters.model_type}')

                if apply_input_delevel:
                    models[horizon] = model.build(model_input, input_mask=input_mask) + input_level
                else:
                    models[horizon] = model.build(model_input, input_mask=input_mask)
    return models


def checkpoints_path(experiment_path: str):
    return os.path.join(experiment_path, 'checkpoints')


def train(experiment_path: str):
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

        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(shape=(batch_size, M4_INPUT_MAXSIZE),
                                    name='inputs',
                                    dtype=tf.float32)
            input_masks = tf.placeholder(shape=(batch_size, M4_INPUT_MAXSIZE),
                                         name='input_masks',
                                         dtype=tf.float32)
            masep = tf.placeholder(shape=(batch_size,),
                                   name='masep',
                                   dtype=tf.float32)
            for horizon in training_set.info.horizons:
                targets[horizon] = tf.placeholder(shape=(batch_size, horizon),
                                                  name=f'target_{horizon}',
                                                  dtype=tf.float32)
                target_masks[horizon] = tf.placeholder(shape=(batch_size, horizon),
                                                       name=f'targets_mask_{horizon}',
                                                       dtype=tf.float32)

        models = model_graph(inputs, input_masks, experiment, training_set.info.horizons)

        # Training operations
        training_operations = {}
        losses = {}
        for horizon in training_set.info.horizons:
            with tf.variable_scope(f'loss_horizon_{horizon}', reuse=False):
                forecast = models[horizon]
                target = targets[horizon]
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
                    # stop gradient on denominator for numerical stability.
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
        for horizon in training_set.info.horizons:
            total_loss = tf.add_n([losses[horizon]] + regularization_losses)
            training_operations[horizon] = slim.learning.create_train_op(total_loss=total_loss,
                                                                         optimizer=optimizer,
                                                                         global_step=global_step,
                                                                         clip_gradient_norm=1.0)

        # Training Summary
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge(tf.get_collection('summaries'))
        summary_writer = tf.summary.FileWriter(experiment_path, flush_secs=1)
        train_log_writer = summary_log(checkpoints_path(experiment_path), writer=summary_writer)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        supervisor = tf.train.Supervisor(logdir=checkpoints_path(experiment_path),
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
                    batch = training_set.sample_batch(batch_size=batch_size,
                                                      indices_filter=experiment.timeseries_indices)
                    feed_dict = {
                        inputs: batch.inputs,
                        input_masks: batch.input_masks,
                        targets[batch.horizon]: batch.targets,
                        target_masks[batch.horizon]: batch.target_mask,
                        masep: batch.masep
                    }
                    batch_loss = sess.run(training_operations[batch.horizon], feed_dict=feed_dict)
                    train_log_results[f'train_loss/horizon_{batch.horizon}'] = batch_loss

                    if step % experiment.parameters.training_checkpoint_interval == 0:
                        train_log_writer(step, **train_log_results)
                        print(f'step {step}, loss: {batch_loss}', flush=True)
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                        saver.save(sess, os.path.join(checkpoints_path(experiment_path), 'model'), global_step=step)


def predict(experiment_path: str):
    experiment = M4Experiment.load(experiment_path)
    training_set = M4Dataset(split=M4DatasetSplit[experiment.parameters.training_split.upper()])

    with tf.Graph().as_default():
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(shape=(None, M4_INPUT_MAXSIZE),
                                    name='inputs',
                                    dtype=tf.float32)
            input_masks = tf.placeholder(shape=(None, M4_INPUT_MAXSIZE),
                                         name='input_masks',
                                         dtype=tf.float32)
        models = model_graph(inputs, input_masks, experiment, training_set.info.horizons)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        session = tf.Session(config=config)
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoints_path(experiment_path))
        saver.restore(session, latest_checkpoint)

        forecasts = []
        for horizon in training_set.info.horizons:
            for batch, batch_mask in training_set.sequential_input_batches(1000, horizon):
                forecasts.extend(session.run(models[horizon], feed_dict={inputs: batch, input_masks: batch_mask}))

        forecasts_df = pd.DataFrame(forecasts)
        forecasts_df.columns = [f'F{i}' for i in range(1, len(forecasts_df.columns) + 1)]
        forecasts_df.index = training_set.info.ids
        forecasts_df.index.name = 'id'
        forecasts_df.to_csv(os.path.join(experiment_path, M4_PREDICTION_FILE_NAME))
