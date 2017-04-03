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

"""Measures FLOPS in convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils


def conv2d(inputs, num_outputs, kernel_size, *args, **kwargs):
  """A wrapper/substitute for conv2d that counts the flops.

  This counts the number of floating-point operations (flops) for a conv2d
  layer, including one with a "mask." The optional keyword argument
  `output_mask` specifies which of the position in the output response map need
  actually be calculated, the rest can be discarded and are not counted in the
  result.

  Since this is a wrapper around slim.conv2d, see that function for details on
  the inputs/outputs.

  Args:
    inputs:      The input response map to the convolution.
    num_outputs: The number of output channels for the convolution.
    kernel_size: Spatial size of the convolution kernel.
    *args:       Additional position arguments forwarded to slim.conv2d.
    **kwargs:    Additional keyword args forwarded to slim.conv2d.
  Returns:
    outputs:     The result of the convolution from slim.conv2d.
    flops:       The operation count as a scalar integer tensor.
  """
  output_mask = kwargs.pop('output_mask', None)

  outputs = slim.conv2d(inputs, num_outputs, kernel_size, *args, **kwargs)

  if inputs.get_shape().is_fully_defined():
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
  else:
    inputs_shape = tf.to_int64(tf.shape(inputs))
    outputs_shape = tf.to_int64(tf.shape(outputs))
  batch_size = outputs_shape[0]

  num_filters_in = inputs_shape[3]
  kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
  if output_mask is None:
    num_spatial_positions = tf.fill(
        # tf.fill does not support int64 dims :-|
        dims=tf.to_int32(tf.stack([batch_size])),
        value=outputs_shape[1] * outputs_shape[2])
  else:
    num_spatial_positions = tf.reduce_sum(output_mask, [1, 2])
  num_spatial_positions = tf.to_int64(num_spatial_positions)

  num_output_positions = num_spatial_positions * num_outputs
  flops = 2 * num_output_positions * (kernel_h * kernel_w * num_filters_in)

  # The numbers are slightly different than TensorFlow graph_metrics since we
  # ignore biases. We do not try to mimic graph_metrics because it is
  # inconsistent in the treatment of biases (batch_norm makes biases "free").
  return outputs, flops


def conv2d_same(inputs,
                num_outputs,
                kernel_size,
                stride,
                rate=1,
                output_mask=None,
                scope=None):
  """Version of TF-Slim resnet_utils.conv2d_same that uses the flopsometer."""
  if stride == 1:
    return conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=rate,
        padding='SAME',
        output_mask=output_mask,
        scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=stride,
        rate=rate,
        padding='VALID',
        output_mask=output_mask,
        scope=scope)
