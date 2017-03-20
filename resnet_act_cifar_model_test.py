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
from tensorflow.contrib import slim

import resnet_act_utils
import resnet_act_cifar_model as resnet


class ResNetActCifarModelTest(tf.test.TestCase):

  def _runBatch(self, is_training, use_act, num_residual_units=[5]):
    batch_size = 3
    height, width = 32, 32
    num_classes = 10

    with slim.arg_scope(resnet.resnet_arg_scope(is_training=is_training)):
      with self.test_session() as sess:
        images = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = resnet.resnet(
            images,
            num_residual_units=num_residual_units,
            num_classes=num_classes,
            use_act=use_act,
            sact=False)
        if use_act:
          metrics = resnet_act_utils.act_metric_map(end_points, False)
          metrics.update(resnet_act_utils.flops_metric_map(end_points, False))
        else:
          metrics = {}

        # Check that there are no global updates as they break tf.cond.
        self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])

        if is_training:
          labels = tf.random_uniform(
              (batch_size,), maxval=num_classes, dtype=tf.int32)
          one_hot_labels = slim.one_hot_encoding(labels, num_classes)
          tf.losses.softmax_cross_entropy(
              logits, one_hot_labels, label_smoothing=0.1, weights=1.0)
          if use_act:
            resnet_act_utils.add_all_ponder_costs(end_points, weights=1.0)
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

  def testTestNoActResidualUnits(self):
    self._runBatch(
        is_training=False, use_act=False, num_residual_units=[1, 2, 3])

  def testTestAct(self):
    self._runBatch(is_training=False, use_act=True)

  def testFlopsNoAct(self):
    batch_size = 3
    height, width = 32, 32
    num_classes = 10

    with slim.arg_scope(resnet.resnet_arg_scope(is_training=False)):
      with self.test_session() as sess:
        images = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = resnet.resnet(
            images,
            num_residual_units=[18],
            num_classes=num_classes,
            use_act=False,
            sact=False)
        flops = sess.run(end_points['flops'])
        # TF graph_metrics value: 506307850 (0.1% difference)
        expected_flops = 505775360
        self.assertAllEqual(flops, [expected_flops] * 3)


class ResNetConvActCifarModelTest(tf.test.TestCase):

  def _runBatch(self, is_training, kernel_size, resolution):
    batch_size = 3
    height, width = 32, 32
    num_classes = 10

    with slim.arg_scope(
        resnet.resnet_arg_scope(
            is_training=is_training,
            sact_kernel_size=kernel_size,
            sact_resolution=resolution)):
      with self.test_session() as sess:
        images = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = resnet.resnet(
            images,
            num_residual_units=[5] * 3,
            num_classes=num_classes,
            use_act=True,
            sact=True)
        metrics = resnet_act_utils.act_metric_map(end_points, False)
        metrics.update(resnet_act_utils.flops_metric_map(end_points, False))

        # Check that there are no global updates as they break tf.cond.
        self.assertEqual(tf.get_collection(tf.GraphKeys.UPDATE_OPS), [])

        if is_training:
          labels = tf.random_uniform(
              (batch_size,), maxval=num_classes, dtype=tf.int32)
          one_hot_labels = slim.one_hot_encoding(labels, num_classes)
          tf.losses.softmax_cross_entropy(
              logits, one_hot_labels, label_smoothing=0.1, weights=1.0)
          resnet_act_utils.add_all_ponder_costs(end_points, weights=1.0)
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
    self._runBatch(is_training=True, kernel_size=1, resolution=0)

  def testTest(self):
    self._runBatch(is_training=False, kernel_size=1, resolution=0)

  def testTrainKernelSize(self):
    self._runBatch(is_training=True, kernel_size=3, resolution=0)

  def testTestKernelSize(self):
    self._runBatch(is_training=False, kernel_size=3, resolution=0)

  def testTrainResolution(self):
    self._runBatch(is_training=True, kernel_size=1, resolution=2)

  def testTestResolution(self):
    self._runBatch(is_training=False, kernel_size=1, resolution=2)

  def testVisualizationBasic(self):
    batch_size = 7
    height, width = 32, 32
    num_classes = 10
    is_training = False
    num_images = 3
    border = 5

    with slim.arg_scope(resnet.resnet_arg_scope(is_training=is_training)):
      with self.test_session() as sess:
        images = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = resnet.resnet(
            images,
            num_residual_units=[5] * 3,
            num_classes=num_classes,
            use_act=True,
            sact=True)

        vis_ponder = resnet_act_utils.sact_image_heatmap(
            end_points,
            'ponder_cost',
            num_images=num_images,
            alpha=0.75,
            border=border)
        vis_units = resnet_act_utils.sact_image_heatmap(
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
