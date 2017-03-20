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
import resnet_act_imagenet_model
import resnet_act_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('master', '',
                       'Name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/resnet/',
                       'Directory where the model was written to.')

tf.app.flags.DEFINE_string('eval_dir', '/tmp/resnet/',
                       'Directory where the results are saved to.')

tf.app.flags.DEFINE_string('dataset_dir', None, 'Directory with Imagenet data.')

tf.app.flags.DEFINE_integer('eval_interval_secs', 600,
                        'The frequency, in seconds, with which evaluation is run.')

tf.app.flags.DEFINE_integer('num_examples', 50000,
                        'The number of examples to evaluate')

tf.app.flags.DEFINE_string(
    'split_name', 'validation',
    'The name of the train/test split, either \'train\' or \'validation\'.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                      'The decay to use for the moving average.')

tf.app.flags.DEFINE_string(
    'num_layers', '101',
    'Depth of the network to train (50, 101, 152, 200), or number of layers'
    ' in each block (e.g. 3_4_23_3).')

tf.app.flags.DEFINE_bool('use_act', True, 'Use ACT?')

tf.app.flags.DEFINE_bool(
    'conv_act', False,
    'Use spatially ACT? Active only when use_act=True.')

tf.app.flags.DEFINE_integer('conv_act_kernel_size', 3,
                        'Kernel size for spatially ACT.')

tf.app.flags.DEFINE_integer('conv_act_resolution', 0,
                        'Resolution of spatially ACT halting probability.')

tf.app.flags.DEFINE_float('tau', 1.0, 'The value of tau (ponder relative cost).')

tf.app.flags.DEFINE_bool('evaluate_once', False, 'Evaluate the model just once?')


def main(_):
  g = tf.Graph()
  with g.as_default():
    tf_global_step = slim.get_or_create_global_step()

    data_tuple = imagenet_data_provider.provide_data(
        FLAGS.split_name,
        FLAGS.batch_size,
        dataset_dir=FLAGS.dataset_dir,
        is_training=False)
    images, one_hot_labels, examples_per_epoch, num_classes = data_tuple

    # Define the model:
    with slim.arg_scope(
        resnet_act_imagenet_model.resnet_arg_scope(
            is_training=False,
            conv_act_kernel_size=FLAGS.conv_act_kernel_size,
            conv_act_resolution=FLAGS.conv_act_resolution)):
      num_layers = resnet_act_utils.parse_num_layers(FLAGS.num_layers)
      logits, end_points = resnet_act_imagenet_model.get_network(
          images,
          num_layers,
          num_classes,
          use_act=FLAGS.use_act,
          conv_act=FLAGS.conv_act)

      # For eval, explicitly add moving_mean and moving_variance variables to
      # the MOVING_AVERAGE_VARIABLES collection.
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)

      for var in tf.get_collection('moving_vars'):
        tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
      for var in slim.get_model_variables():
        tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)

      variables_to_restore = variable_averages.variables_to_restore()
      variables_to_restore[tf_global_step.op.name] = tf_global_step

      predictions = tf.argmax(end_points['predictions'], 1)

      # Define the metrics:
      labels = tf.argmax(one_hot_labels, 1)
      metric_map = {
          'eval/Accuracy':
              slim.metrics.streaming_accuracy(predictions, labels),
          'eval/Recall@5':
              slim.metrics.streaming_recall_at_k(end_points['predictions'],
                                                 labels, 5),
      }
      metric_map.update(resnet_act_utils.flops_metric_map(end_points, True))
      if FLAGS.use_act:
        metric_map.update(resnet_act_utils.act_metric_map(end_points, True))

      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
          metric_map)

      for name, value in names_to_values.iteritems():
        summ = tf.summary.scalar(name, value, collections=[])
        summ = tf.Print(summ, [value], name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, summ)

      if FLAGS.use_act and FLAGS.conv_act:
        resnet_act_utils.add_heatmaps_image_summary(end_points, border=10)

      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))

      if not FLAGS.evaluate_once:
        eval_function = slim.evaluation.evaluation_loop
        checkpoint_path = FLAGS.checkpoint_dir
        kwargs = {'eval_interval_secs': FLAGS.eval_interval_secs}
      else:
        eval_function = slim.evaluation.evaluate_once
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        kwargs = {}

      eval_function(
          FLAGS.master,
          checkpoint_path,
          logdir=FLAGS.eval_dir,
          num_evals=num_batches,
          eval_op=names_to_updates.values(),
          variables_to_restore=variables_to_restore,
          **kwargs)


if __name__ == '__main__':
  tf.app.run()
