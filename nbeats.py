from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


class NBeatsBlock:
    """
    Implementation of N-BEATS block.
    """

    def __init__(self,
                 input_size: int,
                 hidden_units: int,
                 layers: int,
                 forecast_horizon: int,
                 activation_fn=tf.nn.relu,
                 regularizer=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.layers = layers
        self.forecast_horizon = forecast_horizon
        self.activation_fn = activation_fn
        self.regularizer = regularizer

    def build(self, inputs):
        inputs = self.fc(inputs)
        return self.basis(inputs)

    def fc(self, inputs):
        """
        :param inputs: Input tensor of shape (Batch, InputWindow)
        :return: Tensor of shape (Batch, HiddenUnits)
        """
        for i in range(self.layers):
            inputs = slim.fully_connected(inputs=inputs,
                                          num_outputs=self.hidden_units,
                                          weights_regularizer=self.regularizer,
                                          activation_fn=self.activation_fn,
                                          scope=f'fc_{i}')
        return inputs

    def basis(self, inputs):
        """
        Non-parametric basis.

        :param inputs: Input tensor of shape (Batch, HiddenUnits)
        :return: Backcast and forecast tensors.
        """
        basis_output = slim.fully_connected(inputs=inputs,
                                            num_outputs=int(self.forecast_horizon + self.input_size),
                                            weights_regularizer=self.regularizer,
                                            activation_fn=None,
                                            scope='basis_output')
        return basis_output[:, self.forecast_horizon:], basis_output[:, :self.forecast_horizon]


class NBeatsPolynomialBlock(NBeatsBlock):
    def __init__(self,
                 input_size: int,
                 hidden_units: int,
                 layers: int,
                 polynomial_order: int,
                 forecast_horizon: int,
                 activation_fn=tf.nn.relu,
                 regularizer=None):
        super().__init__(input_size, hidden_units, layers, forecast_horizon, activation_fn, regularizer)
        self.polynomial_order = polynomial_order

    def basis(self, inputs):
        """
        Polynomial basis.

        :param inputs: Input tensor of shape (Batch, HiddenUnits)
        :return: Backcast and forecast tensors.
        """
        basis_output = slim.fully_connected(inputs,
                                            num_outputs=2 * (self.polynomial_order + 1),
                                            weights_regularizer=self.regularizer,
                                            activation_fn=None,
                                            scope='polynomial_basis_output')
        # Produce forecast trend synthesis
        t = np.arange(self.forecast_horizon, dtype=np.float32)[None, ] / self.forecast_horizon
        forecast = basis_output[:, 0, None]
        for i in range(1, self.polynomial_order + 1):
            forecast += tf.pow(t, float(i)) * basis_output[:, i, None]
        # Produce backcast trend synthesis, this signal will be used by the next analysis step
        t_backcast = np.arange(self.input_size, dtype=np.float32)[None, ] / self.input_size
        backcast = basis_output[:, self.polynomial_order + 1, None]
        for i in range(1, self.polynomial_order + 1):
            backcast += tf.pow(t_backcast, float(i)) * basis_output[:, self.polynomial_order + 1 + i, None]
        return backcast, forecast


class NBeatsHarmonicsBlock(NBeatsBlock):
    def __init__(self,
                 input_size: int,
                 hidden_units: int,
                 layers: int,
                 num_of_harmonics: int,
                 forecast_horizon: int,
                 activation_fn=tf.nn.relu,
                 regularizer=None):
        super().__init__(input_size, hidden_units, layers, forecast_horizon, activation_fn, regularizer)
        self.num_of_harmonics = num_of_harmonics

    def basis(self, inputs):
        """
        Basis to model seasonality.

        :param inputs: Input tensor of shape (Batch, HiddenUnits)
        :return: Backast and forecast tensors.
        """
        num_basis_fns = int(np.ceil(self.num_of_harmonics / 2 * self.forecast_horizon) - (self.num_of_harmonics - 1))
        harmonics_weights = slim.fully_connected(inputs=inputs,
                                                 num_outputs=4 * num_basis_fns,
                                                 weights_regularizer=self.regularizer,
                                                 activation_fn=None,
                                                 scope='harmonics_weights')
        # Produce forecast seasonality synthesis
        t = np.arange(self.forecast_horizon, dtype=np.float32)[:, None] / self.forecast_horizon
        freq = np.arange(self.num_of_harmonics, self.num_of_harmonics / 2 * self.forecast_horizon,
                         dtype=np.float32) / self.num_of_harmonics
        freq = np.append(np.zeros(1, dtype=np.float32), freq)
        freq = freq[None, :]
        cos_template = np.transpose(np.cos(2 * np.pi * t * freq))
        sin_template = np.transpose(np.sin(2 * np.pi * t * freq))
        harmonics_cos = tf.matmul(harmonics_weights[:, :num_basis_fns], cos_template)
        harmonics_sin = tf.matmul(harmonics_weights[:, num_basis_fns:2 * num_basis_fns], sin_template)
        forecast = harmonics_cos + harmonics_sin
        # Produce backcast seasonality synthesis, this signal will be used by the next analysis step
        # TODO: clarify why /forecast_horizon and not /input_size
        t_backcast = -np.arange(self.input_size, dtype=np.float32)[:, None] / self.forecast_horizon
        cos_template_backcast = np.transpose(np.cos(2 * np.pi * t_backcast * freq))
        sin_template_backcast = np.transpose(np.sin(2 * np.pi * t_backcast * freq))
        cos_backcast = tf.matmul(harmonics_weights[:, 2 * num_basis_fns:3 * num_basis_fns],
                                 cos_template_backcast)
        sin_backcast = tf.matmul(harmonics_weights[:, 3 * num_basis_fns:], sin_template_backcast)
        backcast = cos_backcast + sin_backcast

        return backcast, forecast


class NBeatsStack:
    """
    Implementation of N-BEATS stack.
    """

    def __init__(self, blocks: List[NBeatsBlock]):
        self.blocks = blocks

    def build(self, inputs):
        with tf.variable_scope('nbeats-blocks', reuse=tf.AUTO_REUSE):
            residuals = inputs
            stack_forecast = []
            for block in self.blocks:
                # TODO: fix 0-like mask, it works for M4 but is not generally applicable.
                mask = tf.not_equal(residuals, tf.zeros_like(residuals))
                backcast, forecast = block.build(residuals)
                residuals = residuals - backcast
                stack_forecast.append(forecast)

                residuals = tf.multiply(residuals, tf.cast(mask, residuals.dtype))
        return residuals, tf.add_n(stack_forecast) if len(stack_forecast) > 0 else 0.0


class NBeats:
    def __init__(self,
                 stacks: List[NBeatsStack]):
        self.stacks = stacks

    def build(self, inputs):
        residuals = inputs
        stacks_forecast = 0.0
        for i, stack in enumerate(self.stacks):
            with tf.variable_scope(f'nbeats-stack-{i}', reuse=tf.AUTO_REUSE):
                residuals, forecast = stack.build(residuals)
                stacks_forecast = stacks_forecast + forecast
        return inputs[:, :1] + stacks_forecast
