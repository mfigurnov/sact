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

"""Script to generate random data of the same format as ImageNet.

Creates TFRecord files with the same fields as
tensorflow/models/inception/inception/build_imagenet_data
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

from inception.inception.data import build_imagenet_data


tf.flags.DEFINE_string('out_directory', 'testdata/imagenet',
                       'Output directory for the test data.')

FLAGS = tf.flags.FLAGS


def _random_bounds(n):
  x1, x2 = tuple(np.random.randint(n + 1, size=(2,)) / n)
  return min(x1, x2), max(x1, x2)


def _random_bbox(image_width, image_height):
  xmin, xmax = _random_bounds(image_width)
  ymin, ymax = _random_bounds(image_height)
  return [xmin, ymin, xmax, ymax]


def create_fake_data(split_name, image_width=640, image_height=480):
  """Generates the fake data for a given ImageNet split.

  Args:
    split_name: One of 'train' or 'valdiation'.
    image_width: The width of the random image to generate and write as an
                 integer.
    image_height: Integer height o fthe random image.
  """
  filename = '/tmp/fake_%s.jpg' % split_name

  image = np.random.randint(256, size=(image_height, image_width, 3),
                            dtype=np.uint8)
  image = Image.fromarray(image)
  image_buffer = StringIO.StringIO()
  image.save(image_buffer, format='jpeg')
  image_buffer = image_buffer.getvalue()

  bboxes = [_random_bbox(image_width, image_height)]

  output_file = os.path.join(FLAGS.out_directory,
                             '%s-00000-of-00001' % split_name)
  writer = tf.python_io.TFRecordWriter(output_file)
  # pylint: disable=protected-access
  example = build_imagenet_data._convert_to_example(
      filename, image_buffer, 0, 'n02110341', 'dalmation', bboxes,
      image_height, image_width)
  # pylint: enable=protected-access
  writer.write(example.SerializeToString())
  writer.close()


def main(_):
  create_fake_data('train')
  create_fake_data('validation')


if __name__ == '__main__':
  tf.app.run()
