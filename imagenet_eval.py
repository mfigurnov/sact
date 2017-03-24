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

"""Evaluates a trained ResNet model."""

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

tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'The number of examples to evaluate per evaluation iteration.')

tf.app.flags.DEFINE_string(
    'split_name', 'validation',
    'The name of the train/test split, either \'train\' or \'validation\'.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.')

tf.app.flags.DEFINE_integer('image_size', 224,
                            'Image resolution for resize.')

tf.app.flags.DEFINE_string(
    'model', '101',
    'Depth of the network to train (50, 101, 152, 200), or number of layers'
    ' in each block (e.g. 3_4_23_3).')

tf.app.flags.DEFINE_string(
    'model_type', 'vanilla',
    'Options: vanilla (basic ResNet model), act (Adaptive Computation Time), '
    'act_early_stopping (act implementation which actually saves time), '
    'sact (Spatially Adaptive Computation Time)')

tf.app.flags.DEFINE_float('tau', 1.0, 'The value of tau (ponder relative cost).')

tf.app.flags.DEFINE_bool('evaluate_once', False, 'Evaluate the model just once?')


def main(_):
  g = tf.Graph()
  with g.as_default():
    data_tuple = imagenet_data_provider.provide_data(
        FLAGS.split_name,
        FLAGS.batch_size,
        dataset_dir=FLAGS.dataset_dir,
        is_training=False,
        image_size=FLAGS.image_size)
    images, one_hot_labels, examples_per_epoch, num_classes = data_tuple

    # Define the model:
    with slim.arg_scope(imagenet_model.resnet_arg_scope(is_training=False)):
      model = utils.split_and_int(FLAGS.model)
      logits, end_points = imagenet_model.get_network(
          images,
          model,
          num_classes,
          model_type=FLAGS.model_type)

      predictions = tf.argmax(end_points['predictions'], 1)

      # Define the metrics:
      labels = tf.argmax(one_hot_labels, 1)
      metric_map = {
          'eval/Accuracy':
              tf.contrib.metrics.streaming_accuracy(predictions, labels),
          'eval/Recall@5':
              tf.contrib.metrics.streaming_sparse_recall_at_k(
                  end_points['predictions'], tf.expand_dims(labels, 1), 5),
      }
      metric_map.update(summary_utils.flops_metric_map(end_points, True))
      if FLAGS.model_type in ['act', 'act_early_stopping', 'sact']:
        metric_map.update(summary_utils.act_metric_map(end_points, True))

      names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map(
          metric_map)

      for name, value in names_to_values.iteritems():
        summ = tf.summary.scalar(name, value, collections=[])
        summ = tf.Print(summ, [value], name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, summ)

      if FLAGS.model_type == 'sact':
        summary_utils.add_heatmaps_image_summary(end_points, border=10)

      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))

      if not FLAGS.evaluate_once:
        eval_function = slim.evaluation.evaluation_loop
        checkpoint_path = FLAGS.checkpoint_dir
        kwargs = {'eval_interval_secs': FLAGS.eval_interval_secs}
      else:
        eval_function = slim.evaluation.evaluate_once
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        assert checkpoint_path is not None
        kwargs = {}

      eval_function(
          FLAGS.master,
          checkpoint_path,
          logdir=FLAGS.eval_dir,
          num_evals=num_batches,
          eval_op=names_to_updates.values(),
          **kwargs)


if __name__ == '__main__':
  tf.app.run()
