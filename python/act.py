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

"""Functions for adaptive computation time."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def adaptive_computation_time(halting_proba, eps=1e-2):
  """Gets cost, number of steps and halting dist. for adaptive computation time.

  See Alex Graves "Adaptive Computation Time for Recurrent Neural Networks"
  https://arxiv.org/pdf/1603.08983v4.pdf

  Also see notes by Hugo Larochelle:
  https://www.evernote.com/shard/s189/sh/fd165646-b630-48b7-844c-86ad2f07fcda/c9ab960af967ef847097f21d94b0bff7

  This module makes several assumptions:
  1) The input stays the same for all the timesteps.
  2) The number of possible timesteps for the input
    is limited by `max_timesteps`.
  3) We run all the timesteps for each object during training and inference.
    The unused timesteps are simply "masked".

  Args:
    halting_proba: A 2-D `Tensor` of type `float32`. Probabilities
      of halting the computation at a given timestep for the object.
      Shape is `[batch, max_timesteps - 1]`.
      The values need to be in the range [0, 1].
    eps: A `float` in the range [0, 1]. Small number to ensure that
      the computation can halt after the first timestep.

  Returns:
    ponder_cost: An 1-D `Tensor` of type `float32`.
      A differentiable upper bound on the number of timesteps.
    num_timesteps: An 1-D `Tensor` of type `int32`.
      Actual number of timesteps that took place. num_timesteps < ponder_cost.
    halting_distribution: A 2-D `Tensor` of type `float32`.
      Shape is `[batch, max_timesteps]`. Halting probability distribution.
      halting_distribution[i, j] is probability of computation for i-th object
      to halt at j-th timestep. Sum of every row should be close to one.
  """
  sh = halting_proba.get_shape().as_list()
  batch = sh[0]
  max_timesteps = sh[1] + 1

  zero_col = tf.zeros((batch, 1))

  halting_padded = tf.concat(1, [halting_proba, zero_col])

  halting_cumsum = tf.cumsum(halting_proba, axis=1)
  halting_cumsum_padded = tf.concat(1, [zero_col, halting_cumsum])

  # Does computation halt at this timestep?
  halt_flag = (halting_cumsum >= 1 - eps)
  # Always halt at the final timestep.
  halt_flag_final = tf.concat(1, [halt_flag, tf.fill([batch, 1], True)])

  # Halting iteration (zero-based), eqn. (7).
  # Add a decaying value that ensures that the first true value is selected.
  # The decay value is always less than one.
  decay = 1. / (2. + tf.to_float(tf.range(max_timesteps)))
  halt_flag_final_with_decay = tf.to_float(halt_flag_final) + decay[None, :]
  N = tf.to_int32(tf.argmax(halt_flag_final_with_decay, dimension=1))

  N = tf.stop_gradient(N)

  # Fancy indexing to obtain the value of the remainder. Eqn. (8).
  N_indices = tf.range(batch) * max_timesteps + N
  remainder = 1 - tf.gather(tf.reshape(halting_cumsum_padded, [-1]), N_indices)

  # Switch to one-based indexing here for num_timesteps.
  num_timesteps = N + 1
  ponder_cost = tf.to_float(num_timesteps) + remainder

  timestep_index = tf.range(max_timesteps)[None, :]
  # Calculate the halting distribution, eqn. (6).
  # Fill the first N steps with the halting probabilities.
  # Next values are zero.
  p = tf.select(tf.less(timestep_index, N[:, None]),
                halting_padded,
                tf.zeros((batch, max_timesteps)))
  # Fill the (N+1)-st step with the remainder value.
  p = tf.select(tf.equal(timestep_index, N[:, None]),
                tf.tile(remainder[:, None], tf.pack([1, max_timesteps])),
                p)
  halting_distribution = p

  return (ponder_cost, num_timesteps, halting_distribution)


def run_timesteps(inputs, timestep, max_timesteps, scope, reuse=False):
  """Helper function for running timesteps of the network."""
  states = []
  halting_probas = []
  all_flops = []
  with tf.variable_scope(scope, reuse=reuse):
    state = inputs
    for timestep_idx in range(max_timesteps):
      state, halting_proba, flops = timestep(state, timestep_idx)
      states.append(state)
      halting_probas.append(halting_proba)
      all_flops.append(flops)
  return states, halting_probas, all_flops


def adaptive_computation_time_wrapper(inputs, timestep, max_timesteps,
                                      eps=1e-2, scope='act'):
  """A wrapper of `adaptive_computation_time`.

  Wraps `adaptive_computation_time` with an interface compatible with
  `adaptive_computation_early_stopping`. Should do the same thing as
  `adaptive_computation_early_stopping` but does not use tf.cond and therefore
  works with parameter servers and the GPU.
  """
  states, halting_probas, all_flops = run_timesteps(inputs, timestep,
                                                    max_timesteps, scope)

  (ponder_cost, num_timesteps, halting_distribution) = \
      adaptive_computation_time(tf.concat(1, halting_probas[:-1]))

  if states[0].get_shape().is_fully_defined():
    sh = states[0].get_shape().as_list()
  else:
    sh = tf.shape(states[0])
  batch = sh[0]
  h = tf.reshape(halting_distribution, [batch, 1, max_timesteps])
  s = tf.reshape(tf.pack(states, axis=1), [batch, max_timesteps, -1])
  outputs = tf.batch_matmul(h, s)
  outputs = tf.reshape(outputs, sh)

  flops_per_iteration = [
      f * tf.to_int64(num_timesteps > i) for (i, f) in enumerate(all_flops)
  ]
  flops = tf.add_n(flops_per_iteration)

  return (ponder_cost, num_timesteps, flops, halting_distribution, outputs)


def adaptive_computation_time_conv(inputs, timestep, max_timesteps,
                                   eps=1e-2, scope='act'):
  """Spatially adaptive computation time.

  Each spatial position in the states tensor has its own halting distribution.
  This allows to process different part of an image for a different number of
  timesteps.

  The code is similar to `adaptive_computation_early_stopping`. The differences
  are:
  1) The states are expected to be 4-D tensors (Batch-Height-Width-Channels).
    ACT is applied for first three dimensions.
  2) timestep should have a `residual_mask` argument. It is a `float32` mask
    with 1's corresponding to the positions which need to be updated.
    0's should be frozen. For ResNets this can be achieved by multiplying the
    residual branch responses by `residual_mask`.
  3) There is no tf.cond part so the computation is not actually saved.

  Args:
    inputs: Input states at the first timestep, 4-D `Tensor` of type `float32`.
    timestep: A function. See `adaptive_computation_early_stopping` for
      detailed explanation.
    max_timesteps: Maximum number of timesteps.
    eps: A `float` in the range [0, 1]. Small number to ensure that
      the computation can halt after the first timestep.
    scope: variable scope or scope name in which the layers are created.
      Defaults to 'act'.

  Returns:
    ponder_cost: A 1-D `Tensor` of type `float32`.
      A differentiable upper bound on the number of timesteps.
    num_timesteps: A 1-D `Tensor` of type `int32`.
      Actual number of timesteps that took place. num_timesteps < ponder_cost.
    flops: A 1-D `Tensor` of type `int64`.
      Number of floating point operations that took place.
    halting_distribution: A 2-D `Tensor` of type `float32`.
      Shape is `[batch, max_timesteps]`. Halting probability distribution.
      halting_distribution[i, j] is probability of computation for i-th object
      to halt at j-th timestep. Sum of every row should be close to one.
    outputs: A 4-D `Tensor` of shape [batch, height, width, depth]. Outputs of
      the ACT module, intermediate states weighted by the halting distribution
      for the timesteps.
  """
  with tf.variable_scope(scope):
    halting_distribs = []
    for timestep_idx in range(max_timesteps):

      if not timestep_idx:
        (state, halting_proba, flops) = timestep(
            inputs, timestep_idx, residual_mask=None)

        # Initialize the variables which depend on the state shape.
        state_shape_fully_defined = state.get_shape().is_fully_defined()
        if state_shape_fully_defined:
          sh = state.get_shape().as_list()
          assert len(sh) == 4
        else:
          sh = tf.shape(state)
        halting_cumsum = tf.zeros(sh[:3])
        elements_finished = tf.fill(sh[:3], False)
        remainder = tf.ones(sh[:3])
        # Initialize ponder_cost with one to fix an off-by-one error.
        ponder_cost = tf.ones(sh[:3])
        num_timesteps = tf.zeros(sh[:3], dtype=tf.int32)
      else:
        # Mask out the residual values for the not calculated outputs.
        residual_mask = tf.to_float(tf.logical_not(elements_finished))
        residual_mask = tf.expand_dims(residual_mask, 3)
        (state, halting_proba, current_flops) = timestep(
            state, timestep_idx, residual_mask=residual_mask)
        flops += current_flops

      # We always halt at the last timestep.
      if timestep_idx < max_timesteps - 1:
        halting_proba = tf.reshape(halting_proba, sh[:3])
      else:
        halting_proba = tf.ones(sh[:3])

      halting_cumsum += halting_proba
      # Which objects are no longer calculated after this timestep?
      cur_elements_finished = (halting_cumsum >= 1 - eps)
      # Zero out halting_proba for the previously finished positions.
      halting_proba = tf.select(cur_elements_finished,
                                tf.zeros(sh[:3]),
                                halting_proba)
      # Find positions which have halted at the current timestep.
      just_finished = tf.logical_and(tf.logical_not(elements_finished),
                                     cur_elements_finished)
      # For such positions, the halting distribution value is the remainder.
      # For others, it is the halting_proba.
      cur_halting_distrib = tf.select(just_finished,
                                      remainder,
                                      halting_proba)

      # Update ponder_cost. Add 1 to positions which are still computed,
      # remainder to the positions which have just halted and
      # 0 to the previously halted positions.
      ponder_cost += tf.select(
          cur_elements_finished,
          tf.select(just_finished, remainder, tf.zeros(sh[:3])),
          tf.ones(sh[:3]))

      # Add a timestep to the positions that were active during this timestep
      # (not the ones that will be active the next timestep).
      num_timesteps += tf.to_int32(tf.logical_not(elements_finished))

      # Add new state to the outputs weighted by the halting distribution.
      update = state * tf.expand_dims(cur_halting_distrib, 3)
      if timestep_idx:
        outputs += update
      else:
        outputs = update

      remainder -= halting_proba

      elements_finished = cur_elements_finished

      halting_distribs.append(cur_halting_distrib)

  halting_distribution = tf.pack(halting_distribs, axis=3)

  if not state_shape_fully_defined:
    # Update static shape info. Faster RCNN code wants to know batch dimension
    # statically.
    outputs.set_shape(inputs.get_shape().as_list()[:1] + [None] * 3)

  return (ponder_cost, num_timesteps, flops, halting_distribution, outputs)


def adaptive_computation_early_stopping(inputs, timestep, max_timesteps,
                                        eps=1e-2, scope='act'):
  """Builds adaptive computation module with early stopping of computation.

  `adaptive_computation_time` requires all timesteps to be always
  computed. This function stops the computation as soon as all objects in the
  batch halt. However, if any object still needs calculation, the
  timestep is executed for all objects.

  See `adaptive_computation_time` description for more information.

  Args:
    inputs: Input state at the first timestep. Can have different shape from
      state and output. Should have fully defined shape.
    timestep: A function which is called as follows:
      `new_state, halting_proba, flops = timestep(old_state, timestep_idx)`
      If `timestep_idx==1`, then `old_state` is `inputs`.
      Flops should be a 1-D `Tensor` of length batch_size of type `int64`.
      It can perform different computation depending on `timestep_idx`.
      The function should not have any Python side-effects (due to `tf.cond`
      implementation).

      The function is called two times for each `timestep_idx`.
      1) Outside `tf.cond` to create the necessary variables with reuse=False.
      2) Inside `tf.cond` with reuse=True.
      For this reason, all variables should have static names.
      Good: `w = tf.get_variable('weights', [5, 3])`
      Bad: `w = tf.Variable(tf.zeros([5, 3]))  # The name is auto-generated`
    max_timesteps: Maximum number of timesteps.
    eps: A `float` in the range [0, 1]. Small number to ensure that
      the computation can halt after the first timestep.
    scope: variable scope or scope name in which the layers are created.
      Defaults to 'act'.

  Returns:
    ponder_cost: A 1-D `Tensor` of type `float32`.
      A differentiable upper bound on the number of timesteps.
    num_timesteps: A 1-D `Tensor` of type `int32`.
      Actual number of timesteps that took place. num_timesteps < ponder_cost.
    flops: A 1-D `Tensor` of type `int64`.
      Number of floating point operations that took place.
    halting_distribution: A 2-D `Tensor` of type `float`.
      Shape is `[batch, max_timesteps]`. Halting probability distribution.
      halting_distribution[i, j] is probability of computation for i-th object
      to halt at j-th timestep. Sum of every row should be close to one.
    outputs: A `Tensor` of shape [batch, ...]. Has same shape as states.
      Outputs of the ACT module, intermediate states weighted
      by the halting distribution for the timesteps.
  """
  if inputs.get_shape().is_fully_defined():
    sh = inputs.get_shape().as_list()
  else:
    sh = tf.shape(inputs)
  batch = sh[0]
  inputs_rank = len(sh)

  def _body(timestep_idx, state, halting_cumsum, elements_finished, remainder,
            ponder_cost, num_timesteps, flops, outputs):

    (new_state, halting_proba, cur_flops) = timestep(state, timestep_idx)

    # We always halt at the last timestep.
    if timestep_idx < max_timesteps - 1:
      halting_proba = tf.reshape(halting_proba, [batch])
    else:
      halting_proba = tf.ones([batch])

    halting_cumsum += halting_proba
    cur_elements_finished = (halting_cumsum >= 1 - eps)
    # Zero out halting_proba for the previously finished objects.
    halting_proba = tf.select(cur_elements_finished,
                              tf.zeros([batch]),
                              halting_proba)
    # Find objects which have halted at the current timestep.
    just_finished = tf.logical_and(tf.logical_not(elements_finished),
                                   cur_elements_finished)
    # For such objects, the halting distribution value is the remainder.
    # For others, it is the halting_proba.
    cur_halting_distrib = tf.select(just_finished,
                                    remainder,
                                    halting_proba)

    # Update ponder_cost. Add 1 to objects which are still computed,
    # remainder to the objects which have just halted and
    # 0 to the previously halted objects.
    ponder_cost += tf.select(
        cur_elements_finished,
        tf.select(just_finished, remainder, tf.zeros([batch])),
        tf.ones([batch]))

    # Add a timestep to the objects that were active during this timestep
    # (not the ones that will be active the next timestep).
    evaluated_objects = tf.logical_not(elements_finished)
    num_timesteps += tf.to_int32(evaluated_objects)

    # Update the FLOPS counters for the same objects.
    flops += cur_flops * tf.to_int64(evaluated_objects)

    # Add new state to the outputs weighted by the halting distribution.
    outputs += new_state * tf.reshape(cur_halting_distrib,
                                      [-1] + [1] * (inputs_rank - 1))

    remainder -= halting_proba

    return (new_state, halting_cumsum, cur_elements_finished, remainder,
            ponder_cost, num_timesteps, flops, cur_halting_distrib, outputs)

  def _identity(timestep_idx, state, halting_cumsum, elements_finished,
                remainder, ponder_cost, num_timesteps, flops, outputs):
    return (state, halting_cumsum, elements_finished, remainder, ponder_cost,
            num_timesteps, flops, tf.zeros([batch]), outputs)

  # Create all the variables and losses outside of tf.cond.
  # Without this, regularization losses would not work correctly.
  run_timesteps(inputs, timestep, max_timesteps, scope)

  state = inputs
  halting_cumsum = tf.zeros([batch])
  elements_finished = tf.fill([batch], False)
  remainder = tf.ones([batch])
  # Initialize ponder_cost with one to fix an off-by-one error.
  ponder_cost = tf.ones([batch])
  num_timesteps = tf.zeros([batch], dtype=tf.int32)
  flops = tf.zeros([batch], dtype=tf.int64)

  # We don't know the shape of the outputs. Initialize it to scalar and
  # run the first iteration outside tf.cond (it wants outputs of both
  # branches to have the same shapes).
  outputs = 0.

  # Reuse the variables created above.
  with tf.variable_scope(scope, reuse=True):
    halting_distribs = []
    for timestep_idx in range(max_timesteps):
      finished = tf.reduce_all(elements_finished)
      args = (timestep_idx, state, halting_cumsum, elements_finished, remainder,
              ponder_cost, num_timesteps, flops, outputs)
      if timestep_idx == 0:
        return_values = _body(*args)
      else:
        return_values = tf.cond(finished,
                                lambda: _identity(*args),
                                lambda: _body(*args))
      (state, halting_cumsum, elements_finished, remainder, ponder_cost,
       num_timesteps, flops, cur_halting_distrib, outputs) = return_values

      halting_distribs.append(tf.reshape(cur_halting_distrib, [batch, 1]))

  halting_distribution = tf.concat(1, halting_distribs)

  return (ponder_cost, num_timesteps, flops, halting_distribution, outputs)
