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
    max_units = 8
    h = tf.sigmoid(tf.random_normal(shape=[batch_size, max_units - 1]))
    (cost, num_units, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_units_out, distrib_out) = sess.run(
          (cost, num_units, distrib))
      self.assertEqual(cost_out.shape, (batch_size,))
      self.assertEqual(num_units_out.shape, (batch_size,))
      self.assertEqual(distrib_out.shape, (batch_size, max_units))

  def testEqualValuesInBatch(self):
    batch_size = 2
    max_units = 8
    h = tf.sigmoid(tf.random_normal(shape=[1, max_units - 1]))
    h = tf.tile(h, tf.stack([batch_size, 1]))
    (cost, num_units, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_units_out, distrib_out) = sess.run(
          (cost, num_units, distrib))
      self.assertAlmostEqual(cost_out[0], cost_out[1])
      self.assertEqual(num_units_out[0], num_units_out[1])
      self.assertAllEqual(distrib_out[0], distrib_out[1])

  def testStopsAtFirstUnit(self):
    h = tf.constant([[0.999] * 4])
    (cost, num_units, distrib) = act.adaptive_computation_time(h, eps=1e-2)
    with self.test_session() as sess:
      (cost_out, num_units_out, distrib_out) = sess.run(
          (cost, num_units, distrib))
      self.assertAllClose(cost_out, np.array([2.0]))
      self.assertAllEqual(num_units_out, np.array([1]))
      self.assertAllClose(distrib_out, np.array([[1.] + [0.] * 4]))

  def testStopsAtMiddleUnit(self):
    h = tf.constant([[0.01, 0.50, 0.60, 0.70]])
    (cost, num_units, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_units_out, distrib_out) = sess.run(
          (cost, num_units, distrib))
      self.assertAllClose(cost_out, np.array([3.49]))
      self.assertAllEqual(num_units_out, np.array([3]))
      self.assertAllClose(distrib_out, np.array([[0.01, 0.50, 0.49, 0., 0.]]))

  def testStopsAtLastUnit(self):
    h = tf.constant([[0.01] * 4])
    (cost, num_units, distrib) = act.adaptive_computation_time(h)
    with self.test_session() as sess:
      (cost_out, num_units_out, distrib_out) = sess.run(
          (cost, num_units, distrib))
      self.assertAllClose(cost_out, np.array([5.96]))
      self.assertAllEqual(num_units_out, np.array([5]))
      self.assertAllClose(distrib_out, np.array([[0.01] * 4 + [0.96]]))

  def testCostGradientsStopsAtFirstUnit(self):
    h = tf.constant([[0.999] * 4])
    (cost, num_units, distrib) = act.adaptive_computation_time(h)
    cost_grad = tf.gradients(cost, h)
    with self.test_session() as sess:
      cost_grad_out = sess.run(cost_grad)
      self.assertAllClose(cost_grad_out, np.array([[[0.] * 4]]))

  def testCostGradientsStopsAtMiddleUnit(self):
    h = tf.constant([[0.01, 0.50, 0.60, 0.70]])
    (cost, num_units, distrib) = act.adaptive_computation_time(h)
    cost_grad = tf.gradients(cost, h)
    with self.test_session() as sess:
      cost_grad_out = sess.run(cost_grad)
      self.assertAllClose(cost_grad_out, np.array([[[-1., -1., 0., 0.]]]))

  def testCostGradientsStopsAtLastUnit(self):
    h = tf.constant([[0.01] * 4])
    (cost, num_units, distrib) = act.adaptive_computation_time(h)
    cost_grad = tf.gradients(cost, h)
    with self.test_session() as sess:
      cost_grad_out = sess.run(cost_grad)
      self.assertAllClose(cost_grad_out, np.array([[[-1.] * 4]]))


class ActWrapperTest(tf.test.TestCase):

  def _runAct(self, unit_outputs, halting_probas):
    self.assertEqual(len(unit_outputs), len(halting_probas))
    batch = len(unit_outputs)

    # halting_proba[i][-1] should not be used, but we still pass it here
    # to be able to check that it does not affect anything.
    for (l, h) in zip(unit_outputs, halting_probas):
      self.assertEqual(len(l), len(h))
    max_units = len(unit_outputs[0])

    unit_outputs_tf = tf.constant(
        unit_outputs, shape=[batch, max_units], dtype=tf.float32)
    halting_probas_tf = tf.constant(
        halting_probas, shape=[batch, max_units], dtype=tf.float32)
    # Every unit for each object is two FLOPS.
    flops_tf = tf.constant(2, shape=[batch, max_units], dtype=tf.int64)

    def unit(x, unit_idx):
      return (
          tf.reshape(unit_outputs_tf[:, unit_idx], tf.stack([-1, 1])),
          tf.reshape(halting_probas_tf[:, unit_idx], tf.stack([-1, 1])),
          flops_tf[:, unit_idx])

    inputs = tf.random_normal(shape=[batch, 1])
    (cost, num_units, flops, distrib, outputs
    ) = act.adaptive_computation_time_wrapper(inputs, unit, max_units)
    cost_grad = tf.gradients(cost, halting_probas_tf)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      return sess.run((cost, num_units, flops, distrib, outputs, cost_grad))

  def testEqualValuesInBatch(self):
    (cost, num_units, flops, distrib, outputs, cost_grad) = self._runAct(
        [list(range(5))] * 2, [[0.999] * 5] * 2)
    self.assertAlmostEqual(cost[0], cost[1])
    self.assertEqual(num_units[0], num_units[1])
    self.assertEqual(flops[0], flops[1])
    self.assertAllClose(distrib[0], distrib[1])
    self.assertAllClose(outputs[0], outputs[1])
    self.assertAllClose(cost_grad[0][0], cost_grad[0][1])

  def testStopsAtFirstUnit(self):
    (cost, num_units, flops, distrib, outputs, cost_grad) = self._runAct(
        [list(range(5))], [[0.999] + [0.5] * 4])
    self.assertAllClose(cost, [2.0])
    self.assertAllEqual(num_units, [1])
    self.assertAllEqual(flops, [2])
    self.assertAllClose(distrib, [[1.0] + [0.0] * 4])
    self.assertAllClose(outputs, [[0.0]])
    self.assertAllClose(cost_grad, [[[0.0] * 5]])

  def testStopsAtMiddleUnit(self):
    (cost, num_units, flops, distrib, outputs, cost_grad) = self._runAct(
        [list(range(5))], [[0.01, 0.5, 0.6, 0.7, 0.8]])
    self.assertAllClose(cost, [3.49])
    self.assertAllEqual(num_units, [3])
    self.assertAllEqual(flops, [6])
    self.assertAllClose(distrib, [[0.01, 0.50, 0.49, 0., 0.]])
    self.assertAllClose(outputs, [[1.48]])
    self.assertAllClose(cost_grad, [[[-1., -1., 0., 0., 0.]]])

  def testStopsAtLastUnit(self):
    (cost, num_units, flops, distrib, outputs, cost_grad) = self._runAct(
        [list(range(5))], [[0.01] * 5])
    self.assertAllClose(cost, [5.96])
    self.assertAllEqual(num_units, [5])
    self.assertAllEqual(flops, [10])
    self.assertAllClose(distrib, [[0.01] * 4 + [0.96]])
    self.assertAllClose(outputs, [[3.9]])
    self.assertAllClose(cost_grad, [[[-1.] * 4 + [0.]]])

  def testInputs(self):
    inputs = tf.random_normal(shape=[2, 3])

    def unit(x, unit_idx):
      # First object runs for two units, second object for four units.
      return (x, tf.constant(
          [0.7, 0.3], shape=[2, 1]), tf.constant(
              0, shape=[2], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_time_wrapper(inputs,
                                                                  unit, 5)
    with self.test_session() as sess:
      (inputs_out, outputs_out) = sess.run((inputs, outputs))
      self.assertAllClose(inputs_out, outputs_out)

  def testRegularization(self):
    inputs = tf.random_normal(shape=[1, 3])

    def unit(x, unit_idx):
      with tf.variable_scope('{}'.format(unit_idx)):
        w = tf.get_variable(
            'test_variable', [1, 1],
            initializer=tf.constant_initializer(1.0),
            regularizer=lambda _: 2.0 * tf.nn.l2_loss(_))
      return (w, tf.constant(
          1.0, shape=[1, 1]), tf.constant(
              0, shape=[1], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_time_wrapper(inputs,
                                                                  unit, 5)
    decay_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      (_, decay_cost_out) = sess.run((outputs, decay_cost))
      self.assertEqual(decay_cost_out, 5.0)


class ActEarlyStoppingTest(tf.test.TestCase):

  def _runAct(self, unit_outputs, halting_probas):
    self.assertEqual(len(unit_outputs), len(halting_probas))
    batch = len(unit_outputs)

    # halting_proba[i][-1] should not be used, but we still pass it here
    # to be able to check that it does not affect anything.
    for (l, h) in zip(unit_outputs, halting_probas):
      self.assertEqual(len(l), len(h))
    max_units = len(unit_outputs[0])

    unit_outputs_tf = tf.constant(
        unit_outputs, shape=[batch, max_units], dtype=tf.float32)
    halting_probas_tf = tf.constant(
        halting_probas, shape=[batch, max_units], dtype=tf.float32)
    # Every unit for each object is two FLOPS.
    flops_tf = tf.constant(2, shape=[batch, max_units], dtype=tf.int64)
    unit_counter = tf.Variable(0, trainable=False)

    def unit(x, unit_idx):
      assign_op = unit_counter.assign_add(1)
      with tf.control_dependencies([assign_op]):
        return (
            tf.reshape(unit_outputs_tf[:, unit_idx], tf.stack([-1, 1])),
            tf.reshape(halting_probas_tf[:, unit_idx], tf.stack([-1, 1])),
            flops_tf[:, unit_idx])

    inputs = tf.random_normal(shape=[batch, 1])
    (cost, num_units, flops, distrib, outputs
    ) = act.adaptive_computation_early_stopping(inputs, unit, max_units)
    cost_grad = tf.gradients(cost, halting_probas_tf)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      return sess.run((cost, num_units, flops, distrib, outputs, cost_grad,
                       unit_counter))

  def testEqualValuesInBatch(self):
    (cost, num_units, flops, distrib, outputs, cost_grad,
     unit_counter) = self._runAct([list(range(5))] * 2, [[0.999] * 5] * 2)
    self.assertAlmostEqual(cost[0], cost[1])
    self.assertEqual(num_units[0], num_units[1])
    self.assertEqual(flops[0], flops[1])
    self.assertAllClose(distrib[0], distrib[1])
    self.assertAllClose(outputs[0], outputs[1])
    self.assertAllClose(cost_grad[0][0], cost_grad[0][1])
    self.assertEqual(unit_counter, 1)

  def testStopsAtFirstUnit(self):
    (cost, num_units, flops, distrib, outputs, cost_grad,
     unit_counter) = self._runAct([list(range(5))], [[0.999] + [0.5] * 4])
    self.assertAllClose(cost, [2.0])
    self.assertAllEqual(num_units, [1])
    self.assertAllEqual(flops, [2])
    self.assertAllClose(distrib, [[1.0] + [0.0] * 4])
    self.assertAllClose(outputs, [[0.0]])
    self.assertAllClose(cost_grad, [[[0.0] * 5]])
    self.assertEqual(unit_counter, 1)

  def testStopsAtMiddleUnit(self):
    (cost, num_units, flops, distrib, outputs, cost_grad,
     unit_counter) = self._runAct([list(range(5))], [[0.01, 0.5, 0.6, 0.7, 0.8]])
    self.assertAllClose(cost, [3.49])
    self.assertAllEqual(num_units, [3])
    self.assertAllEqual(flops, [6])
    self.assertAllClose(distrib, [[0.01, 0.50, 0.49, 0., 0.]])
    self.assertAllClose(outputs, [[1.48]])
    self.assertAllClose(cost_grad, [[[-1., -1., 0., 0., 0.]]])
    self.assertEqual(unit_counter, 3)

  def testStopsAtLastUnit(self):
    (cost, num_units, flops, distrib, outputs, cost_grad,
     unit_counter) = self._runAct([list(range(5))], [[0.01] * 5])
    self.assertAllClose(cost, [5.96])
    self.assertAllEqual(num_units, [5])
    self.assertAllEqual(flops, [10])
    self.assertAllClose(distrib, [[0.01] * 4 + [0.96]])
    self.assertAllClose(outputs, [[3.9]])
    self.assertAllClose(cost_grad, [[[-1.] * 4 + [0.]]])
    self.assertEqual(unit_counter, 5)

  def testInputs(self):
    inputs = tf.random_normal(shape=[2, 3])

    def unit(x, unit_idx):
      # First object runs for two units, second object for four units.
      return (x, tf.constant(
          [0.7, 0.3], shape=[2, 1]), tf.constant(
              0, shape=[2], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_early_stopping(inputs,
                                                                    unit, 5)
    with self.test_session() as sess:
      (inputs_out, outputs_out) = sess.run((inputs, outputs))
      self.assertAllClose(inputs_out, outputs_out)

  def testRegularization(self):
    inputs = tf.random_normal(shape=[1, 3])

    def unit(x, unit_idx):
      with tf.variable_scope('{}'.format(unit_idx)):
        w = tf.get_variable(
            'test_variable', [1, 1],
            initializer=tf.constant_initializer(1.0),
            regularizer=lambda _: 2.0 * tf.nn.l2_loss(_))
      return (w, tf.constant(
          1.0, shape=[1, 1]), tf.constant(
              0, shape=[1], dtype=tf.int64))

    (_, _, _, _, outputs) = act.adaptive_computation_early_stopping(inputs,
                                                                    unit, 5)
    decay_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      (outputs_out, decay_cost_out) = sess.run((outputs, decay_cost))
      self.assertEqual(decay_cost_out, 5.0)


class SactTest(tf.test.TestCase):

  def testSimple(self):
    # Batch x Height x Width x Channels
    sh = [1, 1, 2, 1]
    unit_outputs = [
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
    max_units = 3
    residual_masks = []

    def unit(_, unit_idx, residual_mask):
      residual_masks.append(residual_mask)
      return (tf.constant(
          unit_outputs[unit_idx], dtype=tf.float32), tf.constant(
              halting_probas[unit_idx], dtype=tf.float32), tf.constant(
                  flops[unit_idx], shape=[1], dtype=tf.int64))

    inputs = tf.random_normal(shape=sh)
    (cost, num_units, flops, distrib, outputs
    ) = act.spatially_adaptive_computation_time(inputs, unit, max_units)
    with self.test_session() as sess:
      (cost_out, num_units_out, flops_out, distrib_out, outputs_out,
       residual_masks_out) = sess.run(
           (cost, num_units, flops, distrib, outputs, residual_masks[1:]))
    # Batch x Height x Width x Channels
    sh = [1, 1, 2]
    self.assertAllClose(cost_out, np.array([2.1, 3.8]).reshape(sh))
    self.assertAllEqual(num_units_out, np.array([2, 3]).reshape(sh))
    self.assertAllEqual(flops_out, [6])
    distrib_expected = np.array([[0.9, 0.1, 0.0], [0.1, 0.1, 0.8]])
    self.assertAllClose(distrib_out, distrib_expected.reshape(sh + [3]))
    outputs_expected = np.array([1.2, 5.4])
    self.assertAllClose(outputs_out, outputs_expected.reshape(sh + [1]))
    # Residual mask for the second unit
    self.assertAllClose(residual_masks_out[0],
                        np.array([1., 1.]).reshape(sh + [1]))
    # Residual mask for the third unit
    self.assertAllClose(residual_masks_out[1],
                        np.array([0., 1.]).reshape(sh + [1]))

  def testInputs(self):
    max_units = 5
    inputs = tf.random_normal(shape=[2, 5, 3, 3])
    # Generate random probabilities for first four units that sum up to one.
    # Fill in last unit with zeros.
    probas = tf.random_normal(shape=[max_units - 1, 2, 5, 3])
    probas = tf.reshape(probas, [max_units - 1, 2 * 5 * 3])
    probas = tf.nn.softmax(probas)
    probas = tf.reshape(probas, [max_units - 1, 2, 5, 3])
    probas = tf.concat([probas, tf.zeros([1, 2, 5, 3])], 0)

    def unit(x, unit_idx, residual_mask):
      return (x, tf.reshape(probas[unit_idx, :, :, :], [2, 5, 3, 1]),
              tf.zeros(
                  [2], dtype=tf.int64))

    (_, _, _, _, outputs) = act.spatially_adaptive_computation_time(
        inputs, unit, max_units)
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
    max_units = 3

    unit_outputs = []

    def unit(x, unit_idx, residual_mask):
      residual = tf.ones(sh)
      if residual_mask is not None:
        residual *= residual_mask
      outputs = x + residual
      unit_outputs.append(outputs)
      return (outputs, tf.constant(
          halting_probas[unit_idx], dtype=tf.float32), tf.zeros(
              [2], dtype=tf.int64))

    inputs = tf.zeros(sh)
    (_, _, _, _, outputs) = act.spatially_adaptive_computation_time(
        inputs, unit, max_units)
    with self.test_session() as sess:
      unit_outputs_out, final_outputs_out = sess.run(
          (unit_outputs, outputs))

    # First position runs for two iterations,
    # second position for three iterations
    self.assertAllClose(unit_outputs_out[0],
                        np.array([1.0, 1.0]).reshape(sh))
    self.assertAllClose(unit_outputs_out[1],
                        np.array([2.0, 2.0]).reshape(sh))
    self.assertAllClose(unit_outputs_out[2],
                        np.array([2.0, 3.0]).reshape(sh))

    self.assertAllClose(final_outputs_out, np.array([1.1, 2.7]).reshape(sh))


if __name__ == '__main__':
  tf.test.main()
