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

"""Trains a ResNet-ACT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

import imagenet_data_provider
import imagenet_model
import summary_utils
import training_utils
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('master', '',
                       'Name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('train_log_dir', '/tmp/resnet/',
                       'Directory where to write event logs.')

tf.app.flags.DEFINE_string(
    'split_name', 'train',
    """The name of the train/test split, either 'train' or 'validation'.""")

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer('save_interval_secs', 600,
                       'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer('startup_delay_steps', 15,
                       'Number of training steps between replicas startup.')

tf.app.flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')

tf.app.flags.DEFINE_string('dataset_dir', None, 'Directory with ImageNet data.')

# Training parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                        'The number of images in each batch.')

tf.app.flags.DEFINE_float('learning_rate', 0.05, """Initial learning rate.""")

tf.app.flags.DEFINE_float('momentum', 0.9, """Momentum.""")

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                      'Learning rate decay factor.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                      'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

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

tf.app.flags.DEFINE_float('tau', 1.0, 'Target value of tau (ponder relative cost).')

tf.app.flags.DEFINE_string('finetune_path', '',
                       'Path for the initial checkpoint for finetuning.')


def main(_):
  g = tf.Graph()
  with g.as_default():
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    with tf.device(tf.train.replica_device_setter(
        FLAGS.ps_tasks, merge_devices=True)):
      data_tuple = imagenet_data_provider.provide_data(
          FLAGS.split_name,
          FLAGS.batch_size,
          dataset_dir=FLAGS.dataset_dir,
          is_training=True,
          image_size=FLAGS.image_size)
      images, labels, examples_per_epoch, num_classes = data_tuple

      # Define the model:
      with slim.arg_scope(imagenet_model.resnet_arg_scope(is_training=True)):
        model = utils.split_and_int(FLAGS.model)
        logits, end_points = imagenet_model.get_network(
            images,
            model,
            num_classes,
            model_type=FLAGS.model_type)

        # Specify the loss function:
        tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits, label_smoothing=0.1, weights=1.0)
        if FLAGS.model_type in ('act', 'act_early_stopping', 'sact'):
          training_utils.add_all_ponder_costs(end_points, weights=FLAGS.tau)
        total_loss = tf.losses.get_total_loss()

        # Configure the learning rate using an exponetial decay.
        decay_steps = int(examples_per_epoch / FLAGS.batch_size *
                          FLAGS.num_epochs_per_decay)

        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,
            slim.get_or_create_global_step(),
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

        init_fn = training_utils.finetuning_init_fn(FLAGS.finetune_path)

        train_tensor = slim.learning.create_train_op(
            total_loss,
            optimizer=opt,
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        # Summaries:
        tf.summary.scalar('losses/Total Loss', total_loss)
        tf.summary.scalar('training/Learning Rate', learning_rate)

        metric_map = {}  # summary_utils.flops_metric_map(end_points, False)
        if FLAGS.model_type in ('act', 'act_early_stopping', 'sact'):
          metric_map.update(summary_utils.act_metric_map(end_points, False))
        for name, value in metric_map.iteritems():
          tf.summary.scalar(name, value)

        if FLAGS.model_type == 'sact':
          summary_utils.add_heatmaps_image_summary(end_points, border=10)

        startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps

        slim.learning.train(
            train_tensor,
            init_fn=init_fn,
            logdir=FLAGS.train_log_dir,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            startup_delay_steps=startup_delay_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  tf.app.run()
