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

from tensorflow_models.inception.inception import image_processing
from tensorflow_models.slim.datasets import imagenet


def provide_data(split_name, batch_size, dataset_dir=None, is_training=False,
                 num_readers=1, num_preprocess_threads=1, image_size=299):
  """Provides batches of Imagenet data.

  Applies the processing in
    tensorflow/models/inception/inception/image_processing
  to the TF-Slim ImageNet dataset class.

  Args:
    split_name: Either 'train' or 'validation'.
    batch_size: The number of images in each batch.
    dataset_dir: Directory where the ImageNet TFRecord files live.
                 Defaults to "~/tensorflow/data/imagenet"
    is_training: Whether to apply data augmentation and shuffling.
    num_readers: Number of parallel readers.
    num_preprocess_threads: Number of preprocessing threads.

  Returns:
    images: A `Tensor` of size [batch_size, image_size, image_size, 3]
    one_hot_labels: A `Tensor` of size [batch_size, num_classes], where
      each row has a single element set to one and the rest set to zeros.
    dataset.num_samples: The number of total samples in the dataset.
    dataset.num_classes: The number of object classes in the dataset.

  Raises:
    ValueError: if the split_name is not either 'train' or 'validation'.
  """

  if dataset_dir is None:
    dataset_dir = os.path.expanduser('~/tensorflow/data/imagenet')

  dataset = imagenet.get_split(split_name, dataset_dir)
  provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      shuffle=is_training,
      common_queue_capacity=5 * batch_size,
      common_queue_min=batch_size)

  images_and_labels = []
  for thread_id in range(num_preprocess_threads):
    [image, bbox, label] = provider.get(['image', 'object/bbox', 'label'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
      image = image_processing.distort_image(
          image, image_size, image_size, tf.expand_dims(bbox, 0), thread_id)
    else:
      image = image_processing.eval_image(image, image_size, image_size)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    images_and_labels.append([image, label])

  images, labels = tf.train.batch_join(
      images_and_labels,
      batch_size=batch_size,
      capacity=(2 * num_preprocess_threads * batch_size))

  tf.image_summary('images', images)
  one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
  return images, one_hot_labels, dataset.num_samples, dataset.num_classes
