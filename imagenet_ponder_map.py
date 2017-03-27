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

"""Exports ponder cost maps for input images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os

import matplotlib
import matplotlib.image
matplotlib.use('agg')  # disables drawing to X
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import slim

import imagenet_model
import summary_utils
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'model', '101',
    'Depth of the network to train (50, 101, 152, 200), or number of layers'
    ' in each block (e.g. 3_4_23_3).')

tf.app.flags.DEFINE_string('checkpoint_dir', '',
                           'Directory with the checkpoints.')

tf.app.flags.DEFINE_string('images_pattern', '',
                           'Pattern of the JPEG images to process.')

tf.app.flags.DEFINE_string('output_dir', '',
                           'Directory to write the results to.')

tf.app.flags.DEFINE_integer(
    'image_size', 0,
    'Resize the input image so that the longer edge has this many pixels.'
    'Not resizing if set to zero (the default).')

def preprocessing(image):
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def reverse_preprocessing(image):
  image = tf.multiply(image, 0.5)
  image = tf.add(image, 0.5)
  return image


def main(_):
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  num_classes = 1001

  path = tf.placeholder(tf.string)
  contents = tf.read_file(path)
  image = tf.image.decode_jpeg(contents, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  images = tf.expand_dims(image, 0)
  images.set_shape([1, None, None, 3])

  if FLAGS.image_size:
    sh = tf.shape(image)
    height, width = tf.to_float(sh[0]), tf.to_float(sh[1])
    longer_size = tf.constant(FLAGS.image_size, dtype=tf.float32)

    new_size = tf.cond(
      height >= width,
      lambda: (longer_size, (width / height) * longer_size),
      lambda: ((height / width) * longer_size, longer_size))
    images_resized = tf.image.resize_images(images,
        size=tf.to_int32(tf.stack(new_size)),
        method=tf.image.ResizeMethod.BICUBIC)
  else:
    images_resized = images

  images_resized = preprocessing(images_resized)

  # Define the model:
  with slim.arg_scope(imagenet_model.resnet_arg_scope(is_training=False)):
    model = utils.split_and_int(FLAGS.model)
    logits, end_points = imagenet_model.get_network(
        images_resized,
        model,
        num_classes,
        model_type='sact')
    ponder_cost_map = summary_utils.sact_map(end_points, 'ponder_cost')

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  assert checkpoint_path is not None

  saver = tf.train.Saver()
  sess = tf.Session()

  saver.restore(sess, checkpoint_path)

  for current_path in glob.glob(FLAGS.images_pattern):
    print('Processing {}'.format(current_path))

    [image_resized_out, ponder_cost_map_out] = sess.run(
        [tf.squeeze(reverse_preprocessing(images_resized), 0),
         tf.squeeze(ponder_cost_map, [0, 3])],
        feed_dict={path: current_path})

    basename = os.path.splitext(os.path.basename(current_path))[0]
    if FLAGS.image_size:
      matplotlib.image.imsave(
          os.path.join(FLAGS.output_dir, '{}_im.jpg'.format(basename)),
          image_resized_out)
    matplotlib.image.imsave(
        os.path.join(FLAGS.output_dir, '{}_ponder.jpg'.format(basename)),
        ponder_cost_map_out,
        cmap='viridis')

    min_ponder = ponder_cost_map_out.min()
    max_ponder = ponder_cost_map_out.max()
    print('Minimum/maximum ponder cost {:.2f}/{:.2f}'.format(
        min_ponder, max_ponder))

    fig = plt.figure(figsize=(0.2, 2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap='viridis',
        norm=matplotlib.colors.Normalize(vmin=min_ponder, vmax=max_ponder))
    ax.tick_params(labelsize=12)
    filename = os.path.join(FLAGS.output_dir, '{}_colorbar.pdf'.format(basename))
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
  tf.app.run()
