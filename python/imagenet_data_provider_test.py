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

"""Tests for imagenet_data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

import imagenet_data_provider


class ImagenetDataProviderTest(tf.test.TestCase):

  def _testImageNet(self, split_name, expected_num_samples):
    images, one_hot_labels, num_samples, num_classes = \
        imagenet_data_provider.provide_data(split_name, 1,
                                            dataset_dir='testdata/imagenet')
    self.assertEqual(num_samples, expected_num_samples)
    self.assertEqual(num_classes, 1001)
    with self.test_session() as sess:
      with slim.queues.QueueRunners(sess):
        images_out, one_hot_labels_out = sess.run([images, one_hot_labels])
        self.assertEqual(images_out.shape, (1, 299, 299, 3))
        self.assertEqual(one_hot_labels_out.shape, (1, 1001))

  def testImageNetTrainSet(self):
    self._testImageNet('train', 1281167)

  def testImageNetValidationSet(self):
    self._testImageNet('validation', 50000)


if __name__ == '__main__':
  tf.test.main()
