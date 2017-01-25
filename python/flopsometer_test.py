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

"""Tests for flopsometer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import flopsometer


class FlopsometerTest(tf.test.TestCase):

  def testConv2d(self):
    inputs = tf.zeros([2, 16, 16, 4])
    _, flops = flopsometer.conv2d(
        inputs, 8, [3, 3], stride=1, padding='SAME', output_mask=None)
    expected_flops = 2 * 16 * 16 * 3 * 3 * 8 * 4
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      flops_out = sess.run(flops)
      self.assertAllEqual(flops_out, [expected_flops, expected_flops])

  def testConv2dUnknownSize(self):
    inputs = np.zeros([2, 16, 16, 4], dtype=np.float32)
    inputs_tf = tf.placeholder(tf.float32, shape=(2, None, None, 4))
    _, flops = flopsometer.conv2d(
        inputs_tf, 8, [3, 3], stride=1, padding='SAME', output_mask=None)
    expected_flops = 2 * 16 * 16 * 3 * 3 * 8 * 4
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      flops_out = sess.run(flops, feed_dict={inputs_tf: inputs})
      self.assertAllEqual(flops_out, [expected_flops, expected_flops])

  def testConv2dStride(self):
    inputs = tf.zeros([2, 16, 16, 4])
    _, flops = flopsometer.conv2d(
        inputs, 8, [3, 3], stride=2, padding='SAME', output_mask=None)
    output_positions = 8 * 8
    expected_flops = 2 * output_positions * 3 * 3 * 8 * 4
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      flops_out = sess.run(flops)
      self.assertAllEqual(flops_out, [expected_flops, expected_flops])

  def testConv2dOutputMask(self):
    inputs = tf.zeros([2, 16, 16, 4])
    mask = np.random.random([2, 16, 16]) <= 0.6
    mask_tf = tf.constant(np.float32(mask))
    _, flops = flopsometer.conv2d(
        inputs, 8, [3, 3], stride=1, padding='SAME', output_mask=mask_tf)

    per_position_flops = 2 * 3 * 3 * 8 * 4
    num_positions = np.sum(np.sum(np.int32(mask), 2), 1)
    expected_flops = [
        per_position_flops * num_positions[0],
        per_position_flops * num_positions[1]
    ]

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      flops_out = sess.run(flops)
      self.assertAllEqual(flops_out, expected_flops)


if __name__ == '__main__':
  tf.test.main()
