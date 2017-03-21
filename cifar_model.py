# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Adaptive computation time residual network for CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_utils

import flopsometer
import utils


@slim.add_arg_scope
def lrelu(x, leakiness):
  return tf.maximum(x, x * leakiness)


@slim.add_arg_scope
def residual(inputs,
             depth,
             stride,
             activate_before_residual,
             residual_mask=None,
             scope=None):
  with tf.variable_scope(scope, 'residual', [inputs]):
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, scope='preact')
    if activate_before_residual:
      shortcut = preact
    else:
      shortcut = inputs

    if residual_mask is not None:
      # Max-pooling trick only works correctly when stride is 1.
      # We assume that stride=2 happens in the first layer where
      # residual_mask is None.
      assert stride == 1
      diluted_residual_mask = slim.max_pool2d(
          residual_mask, [3, 3], stride=1, padding='SAME')
    else:
      diluted_residual_mask = None

    flops = 0
    conv_output, current_flops = flopsometer.conv2d(
        preact,
        depth,
        3,
        stride=stride,
        padding='SAME',
        output_mask=diluted_residual_mask,
        scope='conv1')
    flops += current_flops

    conv_output, current_flops = flopsometer.conv2d(
        conv_output,
        depth,
        3,
        stride=1,
        padding='SAME',
        activation_fn=None,
        normalizer_fn=None,
        output_mask=residual_mask,
        scope='conv2')
    flops += current_flops

    if depth_in != depth:
      shortcut = slim.avg_pool2d(shortcut, stride, stride, padding='VALID')
      value = (depth - depth_in) // 2
      shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [value, value]])

    if residual_mask is not None:
      conv_output *= residual_mask

    outputs = shortcut + conv_output

    return outputs, flops


def resnet(inputs,
           model,
           num_classes,
           use_act=False,
           sact=False,
           scope='resnet_residual'):
  """Builds a CIFAR-10 resnet model."""
  num_blocks = 3
  num_units = model
  if len(num_units) == 1:
    num_units *= num_blocks
  assert len(num_units) == num_blocks

  b = resnet_utils.Block
  blocks = [
    b('block_1', residual,
      [(16, 1, True)] + [(16, 1, False)] * (num_units[0] - 1)),
    b('block_2', residual,
      [(32, 2, False)] + [(32, 1, False)] * (num_units[1] - 1)),
    b('block_3', residual,
      [(64, 2, False)] + [(64, 1, False)] * (num_units[2] - 1))
  ]

  with tf.variable_scope(scope, [inputs]):
    end_points = {'inputs': inputs}
    end_points['flops'] = 0
    net = inputs
    net, current_flops = flopsometer.conv2d(
        net, 16, 3, activation_fn=None, normalizer_fn=None)
    end_points['flops'] += current_flops
    net, end_points = utils.stack_blocks(
        net,
        blocks,
        use_act=use_act,
        act_early_stopping=True,
        sact=sact,
        end_points=end_points)
    net = tf.reduce_mean(net, [1, 2], keep_dims=True)
    net = slim.batch_norm(net)
    net, current_flops = flopsometer.conv2d(
        net,
        num_classes, [1, 1],
        activation_fn=None,
        normalizer_fn=None,
        scope='logits')
    end_points['flops'] += current_flops
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    return net, end_points


def resnet_arg_scope(is_training=True):
  """Sets up the default arguments for the CIFAR-10 resnet model."""
  batch_norm_params = {
      'is_training':
          is_training,
      'decay':
          0.9,
      'epsilon':
          0.001,
      'scale':
          True,
      # This forces batch_norm to compute the moving averages in-place
      # instead of using a global collection which does not work with tf.cond.
      'updates_collections':
          None,
  }

  with slim.arg_scope([lrelu], leakiness=0.1):
    with slim.arg_scope([slim.conv2d, slim.batch_norm], activation_fn=lrelu):
      with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(0.0002),
          weights_initializer=slim.variance_scaling_initializer(),
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
          return arg_sc
