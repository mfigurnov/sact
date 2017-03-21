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

"""Function for building ResNet with adaptive computation time (ACT)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import h5py
import tensorflow as tf

from tensorflow.contrib import slim

import act
import flopsometer


SACT_KERNEL_SIZE = 3
INIT_BIAS = -3.


def get_halting_proba(outputs):
  with tf.variable_scope('halting_proba'):
    x = outputs
    x = tf.reduce_mean(x, [1, 2], keep_dims=True)

    x = slim.batch_norm(x, scope='global_bn')
    halting_proba, flops = flopsometer.conv2d(
        x,
        1,
        1,
        activation_fn=tf.nn.sigmoid,
        normalizer_fn=None,
        biases_initializer=tf.constant_initializer(INIT_BIAS),
        scope='global_conv')
    halting_proba = tf.squeeze(halting_proba, [1, 2])

    return halting_proba, flops


def get_halting_proba_conv(outputs, residual_mask=None):
  with tf.variable_scope('halting_proba'):
    flops = 0

    x = outputs

    local_feature = slim.batch_norm(x, scope='local_bn')
    halting_logit, current_flops = flopsometer.conv2d(
        local_feature,
        1,
        SACT_KERNEL_SIZE,
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=tf.constant_initializer(INIT_BIAS),
        output_mask=residual_mask,
        scope='local_conv')
    flops += current_flops

    # Add global halting logit.
    global_feature = tf.reduce_mean(x, [1, 2], keep_dims=True)
    global_feature = slim.batch_norm(global_feature, scope='global_bn')
    halting_logit_global, current_flops = flopsometer.conv2d(
        global_feature,
        1,
        1,
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=None,  # biases are already present in local logits
        scope='global_conv')
    flops += current_flops

    # Addition with broadcasting over spatial dimensions.
    halting_logit += halting_logit_global

    halting_proba = tf.sigmoid(halting_logit)

    return halting_proba, flops


def unit_act(block,
              inputs,
              unit_idx,
              skip_halting_proba=False,
              sact=False,
              residual_mask=None):
  with tf.variable_scope('unit_%d' % (unit_idx + 1), [inputs]):
    outputs, flops = block.unit_fn(
        inputs, *block.args[unit_idx], residual_mask=residual_mask)

    if not skip_halting_proba and unit_idx < len(block.args) - 1:
      if sact:
        halting_proba, current_flops = get_halting_proba_conv(outputs)
        flops += current_flops
      else:
        halting_proba, current_flops = get_halting_proba(outputs)
        flops += current_flops
    else:
      halting_proba = None

    return outputs, halting_proba, flops


def stack_blocks(net, blocks, use_act=False, act_early_stopping=False,
                 sact=False, end_points=None):
  """Utility function for assembling SACT models consisting of 'blocks.'"""
  if end_points is None:
    end_points = {}
  end_points['flops'] = end_points.get('flops', 0)
  end_points['block_scopes'] = [block.scope for block in blocks]
  end_points['block_num_units'] = [len(block.args) for block in blocks]

  for block in blocks:
    if use_act:
      if sact:
        act_func = act.spatially_adaptive_computation_time
      else:
        if act_early_stopping:
          act_func = act.adaptive_computation_early_stopping
        else:
          act_func = act.adaptive_computation_time_wrapper

      (ponder_cost, num_units, flops, halting_distribution, net) = act_func(
          net,
          partial(
              unit_act, block, sact=sact),
          len(block.args),
          scope=block.scope)

      end_points['{}/ponder_cost'.format(block.scope)] = ponder_cost
      end_points['{}/num_units'.format(block.scope)] = num_units
      end_points['{}/halting_distribution'.format(
          block.scope)] = halting_distribution
    else:
      with tf.variable_scope(block.scope, 'block', [net]):
        flops = 0
        for unit_idx in range(len(block.args)):
          net, _, current_flops = unit_act(
              block, net, unit_idx, skip_halting_proba=True)
          flops += current_flops

    end_points['{}/flops'.format(block.scope)] = flops
    end_points['flops'] += flops
    end_points[block.scope] = net

  return net, end_points
