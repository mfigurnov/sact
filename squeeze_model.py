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

"""Removes the Momentum and Moving Average variables,
  reducing the model size 2-3 times.
  The provided pretrained models are squeezed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.contrib import slim

import cifar_model
import imagenet_model
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', '/tmp/resnet/',
                           'Directory where the model was written to.')

tf.app.flags.DEFINE_string('output_dir', '/tmp/resnet2/',
                           'Directory where the squeezed model will be written to.')

tf.app.flags.DEFINE_string(
  'model',
  None,
  'A description of the model.')

tf.app.flags.DEFINE_string(
    'model_type', None,
    'Options: vanilla (basic ResNet model), act (Adaptive Computation Time), '
    'act_early_stopping (act implementation which actually saves time), '
    'sact (Spatially Adaptive Computation Time)')

tf.app.flags.DEFINE_string(
    'dataset', None,
    'Options: imagenet, cifar'
)


def main(_):
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  assert FLAGS.model is not None
  assert FLAGS.model_type in ('vanilla', 'act', 'act_early_stopping', 'sact')
  assert FLAGS.dataset in ('imagenet', 'cifar')

  batch_size = 1

  if FLAGS.dataset == 'imagenet':
    height, width = 224, 224
    num_classes = 1001
  elif FLAGS.dataset == 'cifar':
    height, width = 32, 32
    num_classes = 10

  images = tf.random_uniform((batch_size, height, width, 3))
  model = utils.split_and_int(FLAGS.model)

  # Define the model
  if FLAGS.dataset == 'imagenet':
    with slim.arg_scope(imagenet_model.resnet_arg_scope(is_training=False)):
      logits, end_points = imagenet_model.get_network(
          images,
          model,
          num_classes,
          model_type=FLAGS.model_type)
  elif FLAGS.dataset == 'cifar':
    # Define the model:
    with slim.arg_scope(cifar_model.resnet_arg_scope(is_training=False)):
      logits, end_points = cifar_model.resnet(
          images,
          model=model,
          num_classes=num_classes,
          model_type=FLAGS.model_type)

  tf_global_step = slim.get_or_create_global_step()

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.input_dir)
  assert checkpoint_path is not None

  saver = tf.train.Saver(write_version=2)

  with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    saver.save(sess, FLAGS.output_dir + '/model', global_step=tf_global_step)


if __name__ == '__main__':
  tf.app.run()
