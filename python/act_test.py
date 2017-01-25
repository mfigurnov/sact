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

"""Tests for adaptive computation time."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import act


class ActTest(tf.test.TestCase):

  def testOutputSize(self):
    batch_size = 5
    max_timesteps = 8
    h = tf.sigmoid(tf.random_normal(shape=[batch_size, max_timesteps - 1]))
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_timesteps_out, distrib_out) = sess.run(
          (cost, num_timesteps, distrib))
      self.assertEqual(cost_out.shape, (batch_size,))
      self.assertEqual(num_timesteps_out.shape, (batch_size,))
      self.assertEqual(distrib_out.shape, (batch_size, max_timesteps))

  def testEqualValuesInBatch(self):
    batch_size = 2
    max_timesteps = 8
    h = tf.sigmoid(tf.random_normal(shape=[1, max_timesteps - 1]))
    h = tf.tile(h, tf.pack([batch_size, 1]))
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_timesteps_out, distrib_out) = sess.run(
          (cost, num_timesteps, distrib))
      self.assertAlmostEqual(cost_out[0], cost_out[1])
      self.assertEqual(num_timesteps_out[0], num_timesteps_out[1])
      self.assertAllEqual(distrib_out[0], distrib_out[1])

  def testStopsAtFirstTimestep(self):
    h = tf.constant([[0.999] * 4])
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h, eps=1e-2)
    with self.test_session() as sess:
      (cost_out, num_timesteps_out, distrib_out) = sess.run(
          (cost, num_timesteps, distrib))
      self.assertAllClose(cost_out, np.array([2.0]))
      self.assertAllEqual(num_timesteps_out, np.array([1]))
      self.assertAllClose(distrib_out, np.array([[1.] + [0.] * 4]))

  def testStopsAtMiddleTimestep(self):
    h = tf.constant([[0.01, 0.50, 0.60, 0.70]])
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_timesteps_out, distrib_out) = sess.run(
          (cost, num_timesteps, distrib))
      self.assertAllClose(cost_out, np.array([3.49]))
      self.assertAllEqual(num_timesteps_out, np.array([3]))
      self.assertAllClose(distrib_out, np.array([[0.01, 0.50, 0.49, 0., 0.]]))

  def testStopsAtLastTimestep(self):
    h = tf.constant([[0.01] * 4])
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_timesteps_out, distrib_out) = sess.run(
          (cost, num_timesteps, distrib))
      self.assertAllClose(cost_out, np.array([5.96]))
      self.assertAllEqual(num_timesteps_out, np.array([5]))
      self.assertAllClose(distrib_out, np.array([[0.01] * 4 + [0.96]]))

  def testCostGradientsStopsAtFirstTimestep(self):
    h = tf.constant([[0.999] * 4])
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h)
    cost_grad = tf.gradients(cost, h)
    with self.test_session() as sess:
      cost_grad_out = sess.run(cost_grad)
      self.assertAllClose(cost_grad_out, np.array([[[0.] * 4]]))

  def testCostGradientsStopsAtMiddleTimestep(self):
    h = tf.constant([[0.01, 0.50, 0.60, 0.70]])
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h)
    cost_grad = tf.gradients(cost, h)
    with self.test_session() as sess:
      cost_grad_out = sess.run(cost_grad)
      self.assertAllClose(cost_grad_out, np.array([[[-1., -1., 0., 0.]]]))

  def testCostGradientsStopsAtLastTimestep(self):
    h = tf.constant([[0.01] * 4])
    (cost, num_timesteps, distrib) = act.adaptive_computation_time(h)
    cost_grad = tf.gradients(cost, h)
    with self.test_session() as sess:
      cost_grad_out = sess.run(cost_grad)
      self.assertAllClose(cost_grad_out, np.array([[[-1.] * 4]]))


class ActWrapperTest(tf.test.TestCase):

  def _runAct(self, timestep_outputs, halting_probas):
    self.assertEqual(len(timestep_outputs), len(halting_probas))
    batch = len(timestep_outputs)

    # halting_proba[i][-1] should not be used, but we still pass it here
    # to be able to check that it does not affect anything.
    for (l, h) in zip(timestep_outputs, halting_probas):
      self.assertEqual(len(l), len(h))
    max_timesteps = len(timestep_outputs[0])

    timestep_outputs_tf = tf.constant(
        timestep_outputs, shape=[batch, max_timesteps], dtype=tf.float32)
    halting_probas_tf = tf.constant(
        halting_probas, shape=[batch, max_timesteps], dtype=tf.float32)
    # Every timestep for each object is two FLOPS.
    flops_tf = tf.constant(2, shape=[batch, max_timesteps], dtype=tf.int64)

    def timestep(x, timestep_idx):
      return (
          tf.reshape(timestep_outputs_tf[:, timestep_idx], tf.pack([-1, 1])),
          tf.reshape(halting_probas_tf[:, timestep_idx], tf.pack([-1, 1])),
          flops_tf[:, timestep_idx])

    inputs = tf.random_normal(shape=[batch, 1])
    (cost, num_timesteps, flops, distrib, outputs
    ) = act.adaptive_computation_time_wrapper(inputs, timestep, max_timesteps)
    cost_grad = tf.gradients(cost, halting_probas_tf)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      return sess.run((cost, num_timesteps, flops, distrib, outputs, cost_grad))

  def testEqualValuesInBatch(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad) = self._runAct(
        [range(5)] * 2, [[0.999] * 5] * 2)
    self.assertAlmostEqual(cost[0], cost[1])
    self.assertEqual(num_timesteps[0], num_timesteps[1])
    self.assertEqual(flops[0], flops[1])
    self.assertAllClose(distrib[0], distrib[1])
    self.assertAllClose(outputs[0], outputs[1])
    self.assertAllClose(cost_grad[0][0], cost_grad[0][1])

  def testStopsAtFirstTimestep(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad) = self._runAct(
        [range(5)], [[0.999] + [0.5] * 4])
    self.assertAllClose(cost, [2.0])
    self.assertAllEqual(num_timesteps, [1])
    self.assertAllEqual(flops, [2])
    self.assertAllClose(distrib, [[1.0] + [0.0] * 4])
    self.assertAllClose(outputs, [[0.0]])
    self.assertAllClose(cost_grad, [[[0.0] * 5]])

  def testStopsAtMiddleTimestep(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad) = self._runAct(
        [range(5)], [[0.01, 0.5, 0.6, 0.7, 0.8]])
    self.assertAllClose(cost, [3.49])
    self.assertAllEqual(num_timesteps, [3])
    self.assertAllEqual(flops, [6])
    self.assertAllClose(distrib, [[0.01, 0.50, 0.49, 0., 0.]])
    self.assertAllClose(outputs, [[1.48]])
    self.assertAllClose(cost_grad, [[[-1., -1., 0., 0., 0.]]])

  def testStopsAtLastTimestep(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad) = self._runAct(
        [range(5)], [[0.01] * 5])
    self.assertAllClose(cost, [5.96])
    self.assertAllEqual(num_timesteps, [5])
    self.assertAllEqual(flops, [10])
    self.assertAllClose(distrib, [[0.01] * 4 + [0.96]])
    self.assertAllClose(outputs, [[3.9]])
    self.assertAllClose(cost_grad, [[[-1.] * 4 + [0.]]])

  def testInputs(self):
    inputs = tf.random_normal(shape=[2, 3])

    def timestep(x, timestep_idx):
      # First object runs for two timesteps, second object for four timesteps.
      return (x, tf.constant(
          [0.7, 0.3], shape=[2, 1]), tf.constant(
              0, shape=[2], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_time_wrapper(inputs,
                                                                  timestep, 5)
    with self.test_session() as sess:
      (inputs_out, outputs_out) = sess.run((inputs, outputs))
      self.assertAllClose(inputs_out, outputs_out)

  def testRegularization(self):
    inputs = tf.random_normal(shape=[1, 3])

    def timestep(x, timestep_idx):
      with tf.variable_scope('{}'.format(timestep_idx)):
        w = tf.get_variable(
            'test_variable', [1, 1],
            initializer=tf.constant_initializer(1.0),
            regularizer=lambda _: 2.0 * tf.nn.l2_loss(_))
      return (w, tf.constant(
          1.0, shape=[1, 1]), tf.constant(
              0, shape=[1], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_time_wrapper(inputs,
                                                                  timestep, 5)
    decay_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      (_, decay_cost_out) = sess.run((outputs, decay_cost))
      self.assertEqual(decay_cost_out, 5.0)


class ActConvTest(tf.test.TestCase):

  def testSimple(self):
    # Batch x Height x Width x Channels
    sh = [1, 1, 2, 1]
    timestep_outputs = [
        np.array([1.0, 2.0]).reshape(sh),
        np.array([3.0, 4.0]).reshape(sh),
        np.array([5.0, 6.0]).reshape(sh),
    ]
    halting_probas = [
        np.array([0.9, 0.1]).reshape(sh),
        np.array([0.5, 0.1]).reshape(sh),
        np.array([0.8, 0.1]).reshape(sh),  # unused
    ]
    flops = [2, 2, 2]
    max_timesteps = 3
    residual_masks = []

    def timestep(_, timestep_idx, residual_mask):
      residual_masks.append(residual_mask)
      return (tf.constant(
          timestep_outputs[timestep_idx], dtype=tf.float32), tf.constant(
              halting_probas[timestep_idx], dtype=tf.float32), tf.constant(
                  flops[timestep_idx], shape=[1], dtype=tf.int64))

    inputs = tf.random_normal(shape=sh)
    (cost, num_timesteps, flops, distrib, outputs
    ) = act.adaptive_computation_time_conv(inputs, timestep, max_timesteps)
    with self.test_session() as sess:
      (cost_out, num_timesteps_out, flops_out, distrib_out, outputs_out,
       residual_masks_out) = sess.run(
           (cost, num_timesteps, flops, distrib, outputs, residual_masks[1:]))
    # Batch x Height x Width x Channels
    sh = [1, 1, 2]
    self.assertAllClose(cost_out, np.array([2.1, 3.8]).reshape(sh))
    self.assertAllEqual(num_timesteps_out, np.array([2, 3]).reshape(sh))
    self.assertAllEqual(flops_out, [6])
    distrib_expected = np.array([[0.9, 0.1, 0.0], [0.1, 0.1, 0.8]])
    self.assertAllClose(distrib_out, distrib_expected.reshape(sh + [3]))
    outputs_expected = np.array([1.2, 5.4])
    self.assertAllClose(outputs_out, outputs_expected.reshape(sh + [1]))
    # Residual mask for the second timestep
    self.assertAllClose(residual_masks_out[0],
                        np.array([1., 1.]).reshape(sh + [1]))
    # Residual mask for the third timestep
    self.assertAllClose(residual_masks_out[1],
                        np.array([0., 1.]).reshape(sh + [1]))

  def testInputs(self):
    max_timesteps = 5
    inputs = tf.random_normal(shape=[2, 5, 3, 3])
    # Generate random probabilities for first four timesteps that sum up to one.
    # Fill in last timestep with zeros.
    probas = tf.random_normal(shape=[max_timesteps - 1, 2, 5, 3])
    probas = tf.reshape(probas, [max_timesteps - 1, 2 * 5 * 3])
    probas = tf.nn.softmax(probas)
    probas = tf.reshape(probas, [max_timesteps - 1, 2, 5, 3])
    probas = tf.concat(0, [probas, tf.zeros([1, 2, 5, 3])])

    def timestep(x, timestep_idx, residual_mask):
      return (x, tf.reshape(probas[timestep_idx, :, :, :], [2, 5, 3, 1]),
              tf.zeros(
                  [2], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_time_conv(inputs, timestep,
                                                               max_timesteps)
    with self.test_session() as sess:
      (inputs_out, outputs_out) = sess.run((inputs, outputs))
      self.assertAllClose(inputs_out, outputs_out)

  def testResidualMask(self):
    # Batch x Height x Width x Channels
    sh = [1, 1, 2, 1]
    halting_probas = [
        np.array([0.9, 0.1]).reshape(sh),
        np.array([0.5, 0.1]).reshape(sh),
        np.array([0.8, 0.1]).reshape(sh),  # unused
    ]
    max_timesteps = 3

    timestep_outputs = []

    def timestep(x, timestep_idx, residual_mask):
      residual = tf.ones(sh)
      if residual_mask is not None:
        residual *= residual_mask
      outputs = x + residual
      timestep_outputs.append(outputs)
      return (outputs, tf.constant(
          halting_probas[timestep_idx], dtype=tf.float32), tf.zeros(
              [2], dtype=tf.int64))

    inputs = tf.zeros(sh)
    (_, _, _, _, outputs) = act.adaptive_computation_time_conv(inputs, timestep,
                                                               max_timesteps)
    with self.test_session() as sess:
      timestep_outputs_out, final_outputs_out = sess.run(
          (timestep_outputs, outputs))

    # First position runs for two iterations,
    # second position for three iterations
    self.assertAllClose(timestep_outputs_out[0],
                        np.array([1.0, 1.0]).reshape(sh))
    self.assertAllClose(timestep_outputs_out[1],
                        np.array([2.0, 2.0]).reshape(sh))
    self.assertAllClose(timestep_outputs_out[2],
                        np.array([2.0, 3.0]).reshape(sh))

    self.assertAllClose(final_outputs_out, np.array([1.1, 2.7]).reshape(sh))


class ActEarlyStoppingTest(tf.test.TestCase):

  def _runAct(self, timestep_outputs, halting_probas):
    self.assertEqual(len(timestep_outputs), len(halting_probas))
    batch = len(timestep_outputs)

    # halting_proba[i][-1] should not be used, but we still pass it here
    # to be able to check that it does not affect anything.
    for (l, h) in zip(timestep_outputs, halting_probas):
      self.assertEqual(len(l), len(h))
    max_timesteps = len(timestep_outputs[0])

    timestep_outputs_tf = tf.constant(
        timestep_outputs, shape=[batch, max_timesteps], dtype=tf.float32)
    halting_probas_tf = tf.constant(
        halting_probas, shape=[batch, max_timesteps], dtype=tf.float32)
    # Every timestep for each object is two FLOPS.
    flops_tf = tf.constant(2, shape=[batch, max_timesteps], dtype=tf.int64)
    timestep_counter = tf.Variable(0, trainable=False)

    def timestep(x, timestep_idx):
      assign_op = timestep_counter.assign_add(1)
      with tf.control_dependencies([assign_op]):
        return (
            tf.reshape(timestep_outputs_tf[:, timestep_idx], tf.pack([-1, 1])),
            tf.reshape(halting_probas_tf[:, timestep_idx], tf.pack([-1, 1])),
            flops_tf[:, timestep_idx])

    inputs = tf.random_normal(shape=[batch, 1])
    (cost, num_timesteps, flops, distrib, outputs
    ) = act.adaptive_computation_early_stopping(inputs, timestep, max_timesteps)
    cost_grad = tf.gradients(cost, halting_probas_tf)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      return sess.run((cost, num_timesteps, flops, distrib, outputs, cost_grad,
                       timestep_counter))

  def testEqualValuesInBatch(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad,
     timestep_counter) = self._runAct([range(5)] * 2, [[0.999] * 5] * 2)
    self.assertAlmostEqual(cost[0], cost[1])
    self.assertEqual(num_timesteps[0], num_timesteps[1])
    self.assertEqual(flops[0], flops[1])
    self.assertAllClose(distrib[0], distrib[1])
    self.assertAllClose(outputs[0], outputs[1])
    self.assertAllClose(cost_grad[0][0], cost_grad[0][1])
    self.assertEqual(timestep_counter, 1)

  def testStopsAtFirstTimestep(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad,
     timestep_counter) = self._runAct([range(5)], [[0.999] + [0.5] * 4])
    self.assertAllClose(cost, [2.0])
    self.assertAllEqual(num_timesteps, [1])
    self.assertAllEqual(flops, [2])
    self.assertAllClose(distrib, [[1.0] + [0.0] * 4])
    self.assertAllClose(outputs, [[0.0]])
    self.assertAllClose(cost_grad, [[[0.0] * 5]])
    self.assertEqual(timestep_counter, 1)

  def testStopsAtMiddleTimestep(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad,
     timestep_counter) = self._runAct([range(5)], [[0.01, 0.5, 0.6, 0.7, 0.8]])
    self.assertAllClose(cost, [3.49])
    self.assertAllEqual(num_timesteps, [3])
    self.assertAllEqual(flops, [6])
    self.assertAllClose(distrib, [[0.01, 0.50, 0.49, 0., 0.]])
    self.assertAllClose(outputs, [[1.48]])
    self.assertAllClose(cost_grad, [[[-1., -1., 0., 0., 0.]]])
    self.assertEqual(timestep_counter, 3)

  def testStopsAtLastTimestep(self):
    (cost, num_timesteps, flops, distrib, outputs, cost_grad,
     timestep_counter) = self._runAct([range(5)], [[0.01] * 5])
    self.assertAllClose(cost, [5.96])
    self.assertAllEqual(num_timesteps, [5])
    self.assertAllEqual(flops, [10])
    self.assertAllClose(distrib, [[0.01] * 4 + [0.96]])
    self.assertAllClose(outputs, [[3.9]])
    self.assertAllClose(cost_grad, [[[-1.] * 4 + [0.]]])
    self.assertEqual(timestep_counter, 5)

  def testInputs(self):
    inputs = tf.random_normal(shape=[2, 3])

    def timestep(x, timestep_idx):
      # First object runs for two timesteps, second object for four timesteps.
      return (x, tf.constant(
          [0.7, 0.3], shape=[2, 1]), tf.constant(
              0, shape=[2], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_early_stopping(inputs,
                                                                    timestep, 5)
    with self.test_session() as sess:
      (inputs_out, outputs_out) = sess.run((inputs, outputs))
      self.assertAllClose(inputs_out, outputs_out)

  def testRegularization(self):
    inputs = tf.random_normal(shape=[1, 3])

    def timestep(x, timestep_idx):
      with tf.variable_scope('{}'.format(timestep_idx)):
        w = tf.get_variable(
            'test_variable', [1, 1],
            initializer=tf.constant_initializer(1.0),
            regularizer=lambda _: 2.0 * tf.nn.l2_loss(_))
      return (w, tf.constant(
          1.0, shape=[1, 1]), tf.constant(
              0, shape=[1], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_early_stopping(inputs,
                                                                    timestep, 5)
    decay_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      (outputs_out, decay_cost_out) = sess.run((outputs, decay_cost))
      self.assertEqual(decay_cost_out, 5.0)


if __name__ == '__main__':
  tf.test.main()
