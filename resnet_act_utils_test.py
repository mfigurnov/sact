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

"""Tests for resnet_act_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import resnet_act_utils


class ResnetActUtilsTest(tf.test.TestCase):

  def testConvActImageHeatmap(self):
    batch = 9
    num_images = 5
    height, width, channels = 32, 32, 3
    border = 4
    alpha = 0.75

    end_points = {
        'inputs': tf.ones([batch, height, width, channels]),
        'block_num_layers': [10],
        'block_scopes': ['block_1'],
        'block_1/ponder_cost': 5 * tf.ones([batch, height / 2, width / 2]),
    }

    heatmap = resnet_act_utils.sact_image_heatmap(
        end_points,
        'ponder_cost',
        num_images=num_images,
        alpha=alpha,
        border=border,
        normalize_images=False)

    with self.test_session() as sess:
      inputs_out, heatmap_out = sess.run([end_points['inputs'], heatmap])

    self.assertEqual(heatmap_out.shape,
                     (num_images, height, width * 2 + border, channels))
    self.assertAllClose(heatmap_out[:, :, :width, :],
                        inputs_out[:num_images, :, :, :])

    expected_heatmap = 0.25 * inputs_out[:num_images, :, :, :]
    expected_heatmap[:, :, :, 0] += 0.75 * (5.0 / 11.0)
    self.assertAllClose(heatmap_out[:, :, width + border:, :], expected_heatmap)


if __name__ == '__main__':
  tf.test.main()
