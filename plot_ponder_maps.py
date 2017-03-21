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


"""Evaluates a trained ResNet model.

See the README.md file for compilation and running instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import h5py
import matplotlib
matplotlib.use('agg')  # disables drawing to X
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_file', None,
                       'An HDF5 file produced by imagenet_export.')

tf.app.flags.DEFINE_string('output_dir', None,
                       'The directory to output the plotted ponder maps to.')


def main(_):
  f = h5py.File(FLAGS.input_file, 'r')
  num_images = f['images'].shape[0]
  ponder_cost = np.array(f['ponder_cost_map'])
  total_ponder = np.mean(np.reshape(ponder_cost, [num_images, -1]), 1)
  min_ponder = np.percentile(total_ponder, 0.1)
  max_ponder = np.percentile(total_ponder, 99.9)
  sorted_ponder = total_ponder.argsort()

  for idx in range(num_images):
    im = f['images'][idx]
    print(im.shape)
    print(np.amin(im), np.amax(im))
    # Rescales image from [-1, 1] to [0, 1]
    im = (im + 1) / 2

    # Normalizes, and alpha-blends the ponder cost map
    ponder_map = ponder_cost[idx]
    norm_ponder = ((np.clip(ponder_map, min_ponder, max_ponder) - min_ponder)
                 / (max_ponder - min_ponder))
    overlay = im * (0.1 + 0.9 * norm_ponder)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im, vmin=0.0, vmax=1.0, interpolation='nearest')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(overlay, vmin=0.0, vmax=1.0, interpolation='nearest')
    plt.axis('off');
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(ponder_map), vmin=min_ponder, vmax=max_ponder,
               interpolation='nearest', cmap="hot")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off');
    filename = os.path.join(FLAGS.output_dir, 'sact_map_%d.png' % idx)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
  tf.app.run()
