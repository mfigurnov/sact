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

"""Training utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def add_all_ponder_costs(end_points, weights):
  total_ponder_cost = 0.
  for scope in end_points['block_scopes']:
    ponder_cost = end_points['{}/ponder_cost'.format(scope)]
    total_ponder_cost += tf.reduce_mean(ponder_cost)
  tf.losses.add_loss(total_ponder_cost * weights)


def variables_to_str(variables):
  return ', '.join([var.op.name for var in variables])


def finetuning_init_fn(finetune_path):
  """Sets up fine-tuning of a SACT model."""
  if not finetune_path:
    return None

  tf.logging.warning('Finetuning from {}'.format(finetune_path))
  variables = tf.contrib.framework.get_model_variables()
  variables_to_restore = [
      var for var in variables if '/halting_proba/' not in var.op.name
  ]
  tf.logging.info('Restoring variables: {}'.format(
      variables_to_str(variables_to_restore)))
  init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
      finetune_path, variables_to_restore)

  return init_fn
