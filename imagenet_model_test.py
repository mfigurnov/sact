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

"""Tests for imagenet_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import imagenet_model
import summary_utils
import training_utils


class ImagenetModelTest(tf.test.TestCase):

  def _runBatch(self,
                is_training,
                use_act,
                model=[50]):
    batch_size = 3
    height, width = 224, 224
    num_classes = 10

    with self.test_session() as sess:
      images = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope(
          imagenet_model.resnet_arg_scope(is_training=is_training)):
        logits, end_points = imagenet_model.get_network(
            images, model, num_classes, use_act, False)
        if use_act:
          metrics = summary_utils.act_metric_map(end_points, False)
          metrics.update(summary_utils.flops_metric_map(end_points, False))
        else:
          metrics = {}

      # Check that there are no global updates as they break tf.cond.
      # TODO: re-enable
      #self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])

      if is_training:
        labels = tf.random_uniform(
            (batch_size,), maxval=num_classes, dtype=tf.int32)
        one_hot_labels = slim.one_hot_encoding(labels, num_classes)
        tf.losses.softmax_cross_entropy(
            logits, one_hot_labels, label_smoothing=0.1, weights=1.0)
        if use_act:
          training_utils.add_all_ponder_costs(end_points, weights=1.0)
        total_loss = tf.losses.get_total_loss()
        optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        sess.run(tf.global_variables_initializer())
        sess.run((train_op, metrics))
      else:
        sess.run(tf.global_variables_initializer())
        logits_out, metrics_out = sess.run((logits, metrics))
        self.assertEqual(logits_out.shape, (batch_size, num_classes))

  def testTrainNoAct(self):
    self._runBatch(is_training=True, use_act=False)

  def testTrainAct(self):
    self._runBatch(is_training=True, use_act=True)

  def testTestNoAct(self):
    self._runBatch(is_training=False, use_act=False)

  def testTestAct(self):
    self._runBatch(is_training=False, use_act=True)

  def testTestModel(self):
    self._runBatch(is_training=False, use_act=False, model=[3, 4, 6, 3])

  def testFlopsNoAct(self):
    batch_size = 3
    height, width = 224, 224
    num_classes = 1001

    with self.test_session() as sess:
      images = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope(imagenet_model.resnet_arg_scope(is_training=False)):
        _, end_points = imagenet_model.get_network(
            images, [101], num_classes, False, False)
        flops = sess.run(end_points['flops'])
        # TF graph_metrics value: 15614055401 (0.1% difference)
        expected_flops = 15602814976
        self.assertAllEqual(flops, [expected_flops] * 3)


class ResNetSactImagenetModelTest(tf.test.TestCase):

  def _runBatch(self, is_training):
    batch_size = 3
    height, width = 224, 224
    num_classes = 10

    with self.test_session() as sess:
      images = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope(
          imagenet_model.resnet_arg_scope(is_training=is_training)):
        logits, end_points = imagenet_model.get_network(
            images, [50], num_classes, True, True)
        metrics = summary_utils.act_metric_map(end_points, False)
        metrics.update(summary_utils.flops_metric_map(end_points, False))

      # Check that there are no global updates as they break tf.cond.
      # TODO:re-enable
      #self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])

      if is_training:
        labels = tf.random_uniform(
            (batch_size,), maxval=num_classes, dtype=tf.int32)
        one_hot_labels = slim.one_hot_encoding(labels, num_classes)
        tf.losses.softmax_cross_entropy(
            logits, one_hot_labels, label_smoothing=0.1, weights=1.0)
        training_utils.add_all_ponder_costs(end_points, weights=1.0)
        total_loss = tf.losses.get_total_loss()
        optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        sess.run(tf.global_variables_initializer())
        sess.run((train_op, metrics))
      else:
        sess.run(tf.global_variables_initializer())
        logits_out, metrics_out = sess.run((logits, metrics))
        self.assertEqual(logits_out.shape, (batch_size, num_classes))

  def testTrain(self):
    self._runBatch(is_training=True)

  def testTest(self):
    self._runBatch(is_training=False)

  def testVisualizationBasic(self):
    batch_size = 3
    height, width = 224, 224
    num_classes = 10
    is_training = False
    num_images = 3
    border = 5

    with self.test_session() as sess:
      images = tf.random_uniform((batch_size, height, width, 3))
      with slim.arg_scope(imagenet_model.resnet_arg_scope(is_training=is_training)):
        logits, end_points = imagenet_model.get_network(
            images, [50], num_classes, True, True)

        vis_ponder = summary_utils.sact_image_heatmap(
            end_points,
            'ponder_cost',
            num_images=num_images,
            alpha=0.75,
            border=border)
        vis_units = summary_utils.sact_image_heatmap(
            end_points,
            'num_units',
            num_images=num_images,
            alpha=0.75,
            border=border)

        sess.run(tf.global_variables_initializer())
        vis_ponder_out, vis_units_out = sess.run(
            [vis_ponder, vis_units])
        self.assertEqual(vis_ponder_out.shape,
                         (num_images, height, width * 2 + border, 3))
        self.assertEqual(vis_units_out.shape,
                         (num_images, height, width * 2 + border, 3))


if __name__ == '__main__':
  tf.test.main()
