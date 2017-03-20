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

"""Contains code for loading and preprocessing the CIFAR-10 data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim import dataset_data_provider

from external import datasets_cifar10


def provide_data(split_name, batch_size, dataset_dir=None):
  """Provides batches of CIFAR data.

  Args:
    split_name: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    dataset_dir: Directory where the CIFAR-10 TFRecord files live.
                 Defaults to "~/tensorflow/data/cifar10"

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 3]
    images_not_whiten: A `Tensor` with the same size of `images`, unwhitened
      images.
    one_hot_labels: A `Tensor` of size [batch_size, num_classes], where
      each row has a single element set to one and the rest set to zeros.
    dataset.num_samples: The number of total samples in the dataset.
    dataset.num_classes: The number of object classes in the dataset.

  Raises:
    ValueError: if the split_name is not either 'train' or 'test'.
  """
  with tf.device('/cpu:0'):
    is_train = split_name == 'train'

    if dataset_dir is None:
      dataset_dir = os.path.expanduser('~/tensorflow/data/cifar10')

    dataset = datasets_cifar10.get_split(
        split_name, dataset_dir)
    provider = dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=5 * batch_size,
        common_queue_min=batch_size,
        shuffle=is_train)
    [image, label] = provider.get(['image', 'label'])
    image = tf.to_float(image)

    image_size = 32
    if is_train:
      num_threads = 32

      image = tf.image.resize_image_with_crop_or_pad(image, image_size + 4,
                                                     image_size + 4)
      image = tf.random_crop(image, [image_size, image_size, 3])
      image = tf.image.random_flip_left_right(image)
      # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
      # image = tf.image.random_brightness(image, max_delta=63. / 255.)
      # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
      num_threads = 1

      image = tf.image.resize_image_with_crop_or_pad(image, image_size,
                                                     image_size)

    image_not_whiten = image
    image = tf.image.per_image_standardization(image)

    # Creates a QueueRunner for the pre-fetching operation.
    images, images_not_whiten, labels = tf.train.batch(
        [image, image_not_whiten, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)

    labels = tf.reshape(labels, [-1])
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

  return (images, images_not_whiten, one_hot_labels, dataset.num_samples,
          dataset.num_classes)
