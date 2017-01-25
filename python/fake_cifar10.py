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

"""Script to generate random data of the same format as CIFAR-10.

Creates TFRecord files with the same fields as
tensorflow/models/slim/datasets/downlod_and_convert_cifar10.py
for use in unit tests of the code that handles this data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import StringIO

import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow_models.slim.datasets import dataset_utils

tf.flags.DEFINE_string('out_directory', 'testdata/cifar10',
                       'Output directory for the test data.')

FLAGS = tf.flags.FLAGS


_IMAGE_SIZE = 32


def create_fake_data(split_name, num_examples=4):
  """Writes the fake TFRecords for one split of the dataset.

  Args:
    split_name: One of 'train' or 'test'.
    num_examples: The number of random examples to generate and write to the
                  output TFRecord file.
  """
  output_file = os.path.join(FLAGS.out_directory,
                             'cifar10_%s.tfrecord' % split_name)
  writer = tf.python_io.TFRecordWriter(output_file)
  for _ in range(num_examples):
    image = np.random.randint(256, size=(_IMAGE_SIZE, _IMAGE_SIZE, 3),
                              dtype=np.uint8)
    image = Image.fromarray(image)
    image_buffer = StringIO.StringIO()
    image.save(image_buffer, format='png')
    image_buffer = image_buffer.getvalue()

    label = 0
    example = dataset_utils.image_to_tfexample(
        image_buffer, 'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
    writer.write(example.SerializeToString())
  writer.close()


def main(_):
  create_fake_data('train')
  create_fake_data('test')


if __name__ == '__main__':
  tf.app.run()
