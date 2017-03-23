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


"""Draws example ponder cost maps"""

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
  min_ponder = np.percentile(ponder_cost.ravel(), 0.1)
  max_ponder = np.percentile(ponder_cost.ravel(), 99.9)
  print('1st percentile of ponder cost {:.2f} '.format(min_ponder))
  print('99th percentile of ponder cost {:.2f} '.format(max_ponder))

  fig = plt.figure(figsize=(0.2, 2))
  ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
  cb = matplotlib.colorbar.ColorbarBase(
      ax, cmap='viridis',
      norm=matplotlib.colors.Normalize(vmin=min_ponder, vmax=max_ponder))
  ax.tick_params(labelsize=12)
  filename = os.path.join(FLAGS.output_dir, 'colorbar.pdf')
  plt.savefig(filename, bbox_inches='tight')

  for i in range(num_images):
    current_map = np.squeeze(f['ponder_cost_map'][i])
    mean_ponder = np.mean(current_map)
    filename = '{}/{:.2f}_{}_ponder.png'.format(FLAGS.output_dir, mean_ponder, i)
    matplotlib.image.imsave(
        filename, current_map, cmap='viridis', vmin=min_ponder, vmax=max_ponder)

    im = f['images'][i]
    im = (im + 1.0) / 2.0
    filename = '{}/{:.2f}_{}_im.jpg'.format(FLAGS.output_dir, mean_ponder, i)
    matplotlib.image.imsave(filename, im)


if __name__ == '__main__':
  tf.app.run()
