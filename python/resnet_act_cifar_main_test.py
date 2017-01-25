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

"""Tests for resnet_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import resnet_act_cifar_main


FLAGS = tf.flags.FLAGS


class ResNetActCifarMainTest(tf.test.TestCase):

  def testTrain(self):
    FLAGS.batch_size = 4
    FLAGS.max_number_of_steps = 1
    FLAGS.save_interval_secs = 0
    FLAGS.train_log_dir = ''
    FLAGS.num_residual_units = '5'
    FLAGS.use_act = True
    FLAGS.dataset_dir = 'testdata/cifar10'
    resnet_act_cifar_main.train()

  # TODO: re-enable after training a new baseline model
  # def testTrainFinetune(self):
    # FLAGS.batch_size = 4
    # FLAGS.max_number_of_steps = 1
    # FLAGS.save_interval_secs = 0
    # FLAGS.finetune_path = '../models/cifar10_5_act_false_tau_0_v2/train/model.ckpt-100000'
    # FLAGS.train_log_dir = ''
    # FLAGS.num_residitual_units = '5'
    # FLAGS.use_act = True
    # FLAGS.dataset_dir = 'testdata/cifar10'
    # resnet_act_cifar_main.train()


if __name__ == '__main__':
  tf.test.main()
