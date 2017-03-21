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

"""Evaluates a trained ResNet model.

See the README.md file for compilation and running instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.contrib import slim

import imagenet_data_provider
import imagenet_model
import summary_utils
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_examples', 1000, 'The number of examples to evaluate')

tf.app.flags.DEFINE_string(
    'split_name', 'validation',
    'The name of the train/test split, either \'train\' or \'validation\'.')

tf.app.flags.DEFINE_string(
    'model', '101',
    'Depth of the network to train (50, 101, 152, 200), or number of layers'
    ' in each block (e.g. 3_4_23_3).')

tf.app.flags.DEFINE_string(
    'model_type', 'vanilla',
    'Options: vanilla (basic ResNet model), act (Adaptive Computation Time), '
    'act_early_stopping (act implementation which actually saves time), '
    'sact (Spatially Adaptive Computation Time)')

tf.app.flags.DEFINE_string('checkpoint_path', '',
                       'Path for the checkpoint to process.')

tf.app.flags.DEFINE_string('export_path', '',
                       'Path to write the hdf5 file with exported data.')

tf.app.flags.DEFINE_string('dataset_dir', None, 'Directory with Imagenet data.')


def main(_):
  g = tf.Graph()
  with g.as_default():
    tf_global_step = slim.get_or_create_global_step()

    data_tuple = imagenet_data_provider.provide_data(
        FLAGS.split_name,
        FLAGS.batch_size,
        dataset_dir=FLAGS.dataset_dir,
        is_training=False)
    images, labels, _, num_classes = data_tuple

    # Define the model:
    with slim.arg_scope(imagenet_model.resnet_arg_scope(is_training=False)):
      model = utils.split_and_int(FLAGS.model)
      logits, end_points = imagenet_model.get_network(
          images,
          model,
          num_classes,
          model_type=FLAGS.model_type)

      summary_utils.export_to_h5(FLAGS.checkpoint_path, FLAGS.export_path,
                                 images, end_points, FLAGS.num_examples,
                                 FLAGS.batch_size, FLAGS.sact)


if __name__ == '__main__':
  tf.app.run()
