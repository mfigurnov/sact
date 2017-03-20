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

"""Trains a ResNet-ACT model.

See the README.md file for compilation and running instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

import imagenet_data_provider
import resnet_act_imagenet_model
import resnet_act_utils

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

tf.app.flags.DEFINE_bool('sync_replicas', False,
                     'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                     'The decay to use for the moving average.')

tf.app.flags.DEFINE_string(
    'num_layers', '101',
    'Depth of the network to train (50, 101, 152, 200), or number of layers'
    ' in each block (e.g. 3_4_23_3).')

tf.app.flags.DEFINE_bool('use_act', True, 'Use ACT?')

tf.app.flags.DEFINE_bool(
    'sact', False,
    'Use spatially ACT? Active only when use_act=True.')

tf.app.flags.DEFINE_integer('sact_kernel_size', 3,
                        'Kernel size for spatially ACT.')

tf.app.flags.DEFINE_integer('sact_resolution', 0,
                        'Resolution of spatially ACT halting probability.')

tf.app.flags.DEFINE_float('tau', 1.0, 'Target value of tau (ponder relative cost).')

tf.app.flags.DEFINE_float(
    'num_increase_tau_epochs', 0.0,
    'Increase ponder cost penalty from 0 to tau with a linear schedule over'
    ' this many epochs.')

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
          is_training=True)
      images, labels, examples_per_epoch, num_classes = data_tuple

      # Define the model:
      with slim.arg_scope(
          resnet_act_imagenet_model.resnet_arg_scope(
              is_training=True,
              sact_kernel_size=FLAGS.sact_kernel_size,
              sact_resolution=FLAGS.sact_resolution)):
        num_layers = resnet_act_utils.parse_num_layers(FLAGS.num_layers)
        logits, end_points = resnet_act_imagenet_model.get_network(
            images,
            num_layers,
            num_classes,
            use_act=FLAGS.use_act,
            sact=FLAGS.sact)

        # Specify the loss function:
        tf.losses.softmax_cross_entropy(
            logits, labels, label_smoothing=0.1, weights=1.0)
        if FLAGS.use_act:
          # Linear schedule from 0 to tau
          global_step = tf.to_float(slim.get_or_create_global_step())
          tau_target_step = 1.0 * examples_per_epoch / FLAGS.batch_size \
              * FLAGS.num_increase_tau_epochs
          tau = FLAGS.tau * tf.minimum(1.0, global_step / tau_target_step)
          tf.summary.scalar('training/tau', tau)
          resnet_act_utils.add_all_ponder_costs(end_points, weights=tau)
        total_loss = tf.losses.get_total_loss()

        # Setup the moving averages:
        moving_average_variables = slim.get_model_variables()
        moving_average_variables.append(total_loss)

        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, slim.get_or_create_global_step())

        # If sync_replicas is enabled, the averaging will be done in the chief
        # queue runner.
        if not FLAGS.sync_replicas:
          tf.add_to_collection(
              tf.GraphKeys.UPDATE_OPS,
              variable_averages.apply(moving_average_variables))

        # Configure the learning rate using an exponetial decay.
        decay_steps = int(examples_per_epoch / FLAGS.batch_size *
                          FLAGS.num_epochs_per_decay)

        if FLAGS.sync_replicas:
          decay_steps /= FLAGS.replicas_to_aggregate

        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,
            slim.get_or_create_global_step(),
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

        if FLAGS.sync_replicas:
          replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
          opt = tf.train.SyncReplicasOptimizer(
              opt=opt,
              replicas_to_aggregate=FLAGS.replicas_to_aggregate,
              variable_averages=variable_averages,
              variables_to_average=moving_average_variables,
              replica_id=replica_id,
              total_num_replicas=FLAGS.worker_replicas)

        init_fn, gradient_multipliers = \
            resnet_act_utils.get_finetuning_settings(FLAGS.finetune_path)

        train_tensor = slim.learning.create_train_op(
            total_loss,
            optimizer=opt,
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
            gradient_multipliers=gradient_multipliers)

        # Summaries:
        tf.summary.scalar('losses/Total Loss', total_loss)
        tf.summary.scalar('training/Learning Rate', learning_rate)

        metric_map = {}  # resnet_act_utils.flops_metric_map(end_points, False)
        if FLAGS.use_act:
          metric_map.update(resnet_act_utils.act_metric_map(end_points, False))
        for name, value in metric_map.iteritems():
          tf.summary.scalar(name, value)

        if FLAGS.use_act and FLAGS.sact:
          resnet_act_utils.add_heatmaps_image_summary(end_points, border=10)

        if FLAGS.sync_replicas:
          sync_optimizer = opt
          startup_delay_steps = 0
        else:
          sync_optimizer = None
          startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps

        slim.learning.train(
            train_tensor,
            init_fn=init_fn,
            logdir=FLAGS.train_log_dir,
            master=FLAGS.master,
            is_chief=(FLAGS.task == 0),
            startup_delay_steps=startup_delay_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            sync_optimizer=sync_optimizer)


if __name__ == '__main__':
  tf.app.run()
