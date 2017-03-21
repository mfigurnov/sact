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

"""Trains or evaluates a CIFAR ResNet ACT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.contrib import slim

import cifar_data_provider
import resnet_act_cifar_model as resnet
import resnet_act_utils


FLAGS = tf.app.flags.FLAGS

# General settings
tf.app.flags.DEFINE_string('mode', 'train', 'One of "train" or "eval".')

# Training settings
tf.app.flags.DEFINE_integer('batch_size', 128,
                        'The number of images in each batch.')


tf.app.flags.DEFINE_string('master', '',
                       'Name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('train_log_dir', '/tmp/resnet_act_cifar/',
                       'Directory where to write event logs.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 30,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 60,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 100000,
                        'The maximum number of gradient steps.')

tf.app.flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'Directory with CIFAR-10 data, should contain files '
    '"cifar10_train.tfrecord" and "cifar10_test.tfrecord".')

# Evaluation settings
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/resnet_act_cifar/',
                       'Directory where the model was written to.')

tf.app.flags.DEFINE_string('eval_dir', '/tmp/resnet_act_cifar/',
                       'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer('eval_batch_size', 100,
                        'The number of images in each batch for evaluation.')

tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 60,
    'The frequency, in seconds, with which evaluation is run.')

tf.app.flags.DEFINE_string('split_name', 'test', """Either 'train' or 'test'.""")

tf.app.flags.DEFINE_bool('evaluate_once', False, 'Evaluate the model just once?')

# Model settings
tf.app.flags.DEFINE_bool('use_act', False, 'Use ACT?')

tf.app.flags.DEFINE_bool(
    'sact', False,
    'Use spatially ACT? Active only when use_act=True.')

tf.app.flags.DEFINE_integer('sact_kernel_size', 3,
                        'Kernel size for spatially ACT.')

tf.app.flags.DEFINE_integer('sact_resolution', 0,
                        'Resolution of spatially ACT halting probability.')

tf.app.flags.DEFINE_float('tau', 1.0, 'The value of tau (ponder relative cost).')

tf.app.flags.DEFINE_float('learning_rate_mult', 1.0,
                      'Multiplier for learning rate.')

tf.app.flags.DEFINE_string(
  'model',
  '18',
  'An underscore separated string, number of residual units per block. '
  'If only one number is provided, uses the same number of units in all blocks')

tf.app.flags.DEFINE_string('finetune_path', '',
                       'Path for the initial checkpoint for finetuning.')


def get_tau_value(use_schedule=False):
  if use_schedule:
    global_step = tf.to_float(slim.get_or_create_global_step())
    tau_target_step = 30000.0
    tau = FLAGS.tau * tf.minimum(1.0, global_step / tau_target_step)
    return tau
  else:
    return FLAGS.tau


def train():
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    tf.gfile.MakeDirs(FLAGS.train_log_dir)

  g = tf.Graph()
  with g.as_default():
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      data_tuple = cifar_data_provider.provide_data(
          'train', FLAGS.batch_size, dataset_dir=FLAGS.dataset_dir)
      images, _, one_hot_labels, _, num_classes = data_tuple

      # Define the model:
      with slim.arg_scope(
          resnet.resnet_arg_scope(
              is_training=True,
              sact_kernel_size=FLAGS.sact_kernel_size,
              sact_resolution=FLAGS.sact_resolution)):
        model = resnet_act_utils.split_and_int(FLAGS.model)
        logits, end_points = resnet.resnet(
            images,
            model=model,
            num_classes=num_classes,
            use_act=FLAGS.use_act,
            sact=FLAGS.sact)

        # Specify the loss function:
        tf.losses.softmax_cross_entropy(logits, one_hot_labels)
        if FLAGS.use_act:
          tau = get_tau_value()
          tf.summary.scalar('training/tau', tau)
          resnet_act_utils.add_all_ponder_costs(end_points, weights=tau)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('Total Loss', total_loss)

        metric_map = {}  # resnet_act_utils.flops_metric_map(end_points, False)
        if FLAGS.use_act:
          metric_map.update(resnet_act_utils.act_metric_map(end_points, False))
        for name, value in metric_map.iteritems():
          tf.summary.scalar(name, value)

        if FLAGS.use_act and FLAGS.sact:
          resnet_act_utils.add_heatmaps_image_summary(end_points)

        init_fn, _ = resnet_act_utils.get_finetuning_settings(
            FLAGS.finetune_path)

        # Specify the optimization scheme:
        global_step = slim.get_or_create_global_step()
        # Original LR schedule
        # boundaries = [40000, 60000, 80000]
        # "Longer" LR schedule
        boundaries = [60000, 75000, 90000]
        boundaries = [tf.constant(x, dtype=tf.int64) for x in boundaries]
        values = [0.1, 0.01, 0.001, 0.0001]
        if FLAGS.learning_rate_mult != 1.0:
          values = [x * FLAGS.learning_rate_mult for x in values]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries,
                                                    values)
        tf.summary.scalar('Learning Rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

        # Set up training.
        train_op = slim.learning.create_train_op(
            total_loss, optimizer, gradient_multipliers=None)

        if FLAGS.train_log_dir:
          logdir = FLAGS.train_log_dir
        else:
          logdir = None

        # Run training.
        slim.learning.train(
            train_op=train_op,
            init_fn=init_fn,
            logdir=logdir,
            master=FLAGS.master,
            number_of_steps=FLAGS.max_number_of_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs)


def evaluate():
  g = tf.Graph()
  with g.as_default():
    data_tuple = cifar_data_provider.provide_data(FLAGS.split_name,
                                                  FLAGS.eval_batch_size,
                                                  dataset_dir=FLAGS.dataset_dir)
    images, _, one_hot_labels, num_samples, num_classes = data_tuple

    # Define the model:
    with slim.arg_scope(
        resnet.resnet_arg_scope(
            is_training=False,
            sact_kernel_size=FLAGS.sact_kernel_size,
            sact_resolution=FLAGS.sact_resolution)):
      model = resnet_act_utils.split_and_int(FLAGS.model)
      logits, end_points = resnet.resnet(
          images,
          model=model,
          num_classes=num_classes,
          use_act=FLAGS.use_act,
          sact=FLAGS.sact)

      predictions = tf.argmax(logits, 1)

      tf.losses.softmax_cross_entropy(logits, one_hot_labels)
      if FLAGS.use_act:
        tau = get_tau_value()
        resnet_act_utils.add_all_ponder_costs(end_points, weights=tau)

      loss = tf.losses.get_total_loss()

      # Define the metrics:
      labels = tf.argmax(one_hot_labels, 1)
      metric_map = {
          'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
          'eval/Mean Loss': slim.metrics.streaming_mean(loss),
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

      if FLAGS.use_act and FLAGS.sact:
        resnet_act_utils.add_heatmaps_image_summary(end_points)

      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(num_samples / float(FLAGS.eval_batch_size))

      if not FLAGS.evaluate_once:
        eval_function = slim.evaluation.evaluation_loop
        checkpoint_path = FLAGS.checkpoint_dir
        eval_kwargs = {'eval_interval_secs': FLAGS.eval_interval_secs}
      else:
        eval_function = slim.evaluation.evaluate_once
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        assert checkpoint_path is not None
        eval_kwargs = {}

      eval_function(
          FLAGS.master,
          checkpoint_path,
          logdir=FLAGS.eval_dir,
          num_evals=num_batches,
          eval_op=names_to_updates.values(),
          **eval_kwargs)


def main(_):
  if FLAGS.mode == 'train':
    train()
  elif FLAGS.mode == 'eval':
    evaluate()


if __name__ == '__main__':
  tf.app.run()
