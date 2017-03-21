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

"""Definition of Resnet-ACT model used for imagenet classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_utils

import act
import flopsometer
import resnet_act


def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               residual_mask=None,
               scope=None):
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    flops = 0

    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut, current_flops = flopsometer.conv2d(
          preact,
          depth, [1, 1],
          stride=stride,
          normalizer_fn=None,
          activation_fn=None,
          scope='shortcut')
      flops += current_flops

    if residual_mask is not None:
      # Max-pooling trick only works correctly when stride is 1.
      # We assume that stride=2 happens in the first layer where
      # residual_mask is None.
      assert stride == 1
      diluted_residual_mask = slim.max_pool2d(
          residual_mask, [3, 3], stride=1, padding='SAME')
    else:
      diluted_residual_mask = None

    residual, current_flops = flopsometer.conv2d(
        preact,
        depth_bottleneck, [1, 1],
        stride=1,
        output_mask=diluted_residual_mask,
        scope='conv1')
    flops += current_flops

    residual, current_flops = flopsometer.conv2d_same(
        residual,
        depth_bottleneck,
        3,
        stride,
        rate=rate,
        output_mask=residual_mask,
        scope='conv2')
    flops += current_flops

    residual, current_flops = flopsometer.conv2d(
        residual,
        depth, [1, 1],
        stride=1,
        normalizer_fn=None,
        activation_fn=None,
        output_mask=residual_mask,
        scope='conv3')
    flops += current_flops

    if residual_mask is not None:
      residual *= residual_mask

    outputs = shortcut + residual

    return outputs, flops


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              model_type='vanilla',
              scope=None,
              reuse=None,
              end_points=None):
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    if end_points is None:
      end_points = {}
    end_points['inputs'] = inputs
    end_points['flops'] = end_points.get('flops', 0)
    net = inputs
    # We do not include batch normalization or activation functions in conv1
    # because the first ResNet unit will perform these. Cf. Appendix of [2].
    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
      net, current_flops = flopsometer.conv2d_same(
          net, 64, 7, stride=2, scope='conv1')
      end_points['flops'] += current_flops
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
    # Early stopping is broken in distributed training.
    net, end_points = resnet_act.stack_blocks(
        net,
        blocks,
        model_type=model_type,
        end_points=end_points)

    if global_pool or num_classes is not None:
      # This is needed because the pre-activation variant does not have batch
      # normalization or activation functions in the residual unit output. See
      # Appendix of [2].
      net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

    if global_pool:
      # Global average pooling.
      net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    if num_classes is not None:
      net, current_flops = flopsometer.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='logits')
      end_points['flops'] += current_flops
      end_points['predictions'] = slim.softmax(net, scope='predictions')
    return net, end_points


def resnet_arg_scope(is_training=True):
  return resnet_utils.resnet_arg_scope(is_training)
  # with slim.arg_scope(resnet_utils.resnet_arg_scope(is_training)):
    # # This forces batch_norm to compute the moving averages in-place
    # # instead of using a global collection which does not work with tf.cond.
    # with slim.arg_scope([slim.batch_norm], updates_collections=None) as arg_sc:
    #   return arg_sc


def get_network(images,
                model,
                num_classes,
                model_type='vanilla',
                global_pool=True,
                base_channels=64,
                scope=None,
                reuse=None,
                end_points=None):
  # These settings are *not* compatible with Slim's ResNet v2.
  # In ResNet Slim the downsampling is performed by the last layer of the
  # current block. Here we perform downsampling in the first layer of the next
  # block. This is consistent with the ResNet paper.
  num_blocks = 4
  if len(model) == 1:
    standard_networks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }
    num_units = standard_networks[model[0]]
  else:
    num_units = model
  assert len(num_units) == num_blocks

  b = resnet_utils.Block
  bc = base_channels
  blocks = [
      b('block1', bottleneck, [(4 * bc, bc, 1)] * num_units[0]),
      b('block2', bottleneck,
        [(8 * bc, 2 * bc, 2)] + [(8 * bc, 2 * bc, 1)] * (num_units[1] - 1)),
      b('block3', bottleneck,
        [(16 * bc, 4 * bc, 2)] + [(16 * bc, 4 * bc, 1)] * (num_units[2] - 1)),
      b('block4', bottleneck,
        [(32 * bc, 8 * bc, 2)] + [(32 * bc, 8 * bc, 1)] * (num_units[3] - 1)),
  ]

  logits, end_points = resnet_v2(
      images,
      blocks,
      num_classes,
      global_pool=global_pool,
      model_type=model_type,
      scope=scope,
      reuse=reuse,
      end_points=end_points)

  if num_classes is not None and global_pool:
    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    end_points['predictions'] = tf.squeeze(
        end_points['predictions'], [1, 2], name='SpatialSqueeze')
  return logits, end_points
