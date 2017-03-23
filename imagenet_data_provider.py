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

"""Contains code for loading and preprocessing the ImageNet data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim import dataset_data_provider

from external import inception_preprocessing
from external import datasets_imagenet


def provide_data(split_name, batch_size, dataset_dir=None, is_training=False,
                 num_readers=4, num_preprocessing_threads=4, image_size=224):
  """Provides batches of Imagenet data.

  Applies the processing in external/inception_preprocessing
  to the TF-Slim ImageNet dataset class.

  Args:
    split_name: Either 'train' or 'validation'.
    batch_size: The number of images in each batch.
    dataset_dir: Directory where the ImageNet TFRecord files live.
                 Defaults to "~/tensorflow/data/imagenet"
    is_training: Whether to apply data augmentation and shuffling.
    num_readers: Number of parallel readers. Always set to one for evaluation.
    num_preprocessing_threads: Number of preprocessing threads.

  Returns:
    images: A `Tensor` of size [batch_size, image_size, image_size, 3]
    one_hot_labels: A `Tensor` of size [batch_size, num_classes], where
      each row has a single element set to one and the rest set to zeros.
    dataset.num_samples: The number of total samples in the dataset.
    dataset.num_classes: The number of object classes in the dataset.

  Raises:
    ValueError: if the split_name is not either 'train' or 'validation'.
  """

  with tf.device('/cpu:0'):
    if dataset_dir is None:
      dataset_dir = os.path.expanduser('~/tensorflow/data/imagenet')

    if not is_training:
      num_readers = 1

    dataset = datasets_imagenet.get_split(split_name, dataset_dir)
    provider = dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        shuffle=is_training,
        common_queue_capacity=5 * batch_size,
        common_queue_min=batch_size)

    [image, bbox, label] = provider.get(['image', 'object/bbox', 'label'])
    bbox = tf.expand_dims(bbox, 0)

    image = inception_preprocessing.preprocess_image(
      image, image_size, image_size, is_training, bbox, fast_mode=False)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size)

    one_hot_labels = tf.one_hot(labels, dataset.num_classes)

  return images, one_hot_labels, dataset.num_samples, dataset.num_classes
