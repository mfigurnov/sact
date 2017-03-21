Code for "Spatially Adaptive Computation Time for Residual Networks."

https://arxiv.org/abs/1612.02297

This code implements a deep learning architecture based on Residual Network that
dynamically adjusts the number of executed layers for the regions of the image.
The architecture is end-to-end trainable, deterministic and problem-agnostic.
The included code applies this to the CIFAR-10 an ImageNet image classification
problems. It is implemented using TensorFlow and TF-Slim.

SETUP
==========

Prerequite packages:
 - [numpy](http://www.numpy.org/)
 - [tensorflow](https://www.tensorflow.org/):
   [official setup instructions](
   https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
 - [h5py](http://www.h5py.org/):
   [setup instructions](http://docs.h5py.org/en/latest/build.html)
 - [tensorflow/models distribution](https://github.com/tensorflow/models)
 - (Optional) [bazel](https://bazel.build/):
   [setup instructions](https://bazel.build/versions/master/docs/install.html)
 - (Optional) [matplotlib](http://matplotlib.org/):
   [setup instructions](http://matplotlib.org/users/installing.html)

Currently this requires the latest *nightly* release of TensorFlow and some of
the distributions. Setup instructions are at:
https://github.com/tensorflow/models/tree/master/slim.

A full set of sample commands to train and evaluate a simple CIFAR-10 model on Linux, using Python 2 and CPU-only version of TensorFlow:

``` bash
TF_BINARY_URL=https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade "${TF_BINARY_URL}"
```

Set up the tensorflow/models distribution

``` bash
mkdir "${HOME}/tensorflow"
cd "${HOME}/tensorflow"
git clone 'https://github.com/tensorflow/models.git'
cd models/slim
```

Download and convert CIFAR-10 dataset
(see https://github.com/tensorflow/models/tree/master/slim for details)

``` bash
mkdir "${HOME}/tensorflow/data"
python download_and_convert_data.py --dataset_name=cifar10 --dataset_dir="${HOME}/tensorflow/data/cifar10"
```

``` bash
# TODO: below should be replaced with the real URL upon release
cd "${HOME}"
git clone '<url to SACT distribution>' sact
cd sact
```

Note that bazel builds must turn off visibility to use tensorflow/models/slim
code.

``` bash
bazel build :resnet_act_cifar_main --nocheck_visibility
```

Training and continuously evaluating a CIFAR-10 Resnet-ACT model:

``` bash
export ACT_LOGDIR='/tmp/resnet_act_cifar'
bazel-bin/resnet_act_cifar_main --use_act=True --tau=0.01 --train_log_dir="${ACT_LOGDIR}/train" --save_summaries_secs=300 &
bazel-bin/resnet_act_cifar_main --use_act=True --tau=0.01 --checkpoint_dir="${ACT_LOGDIR}/train" --eval_dir="${ACT_LOGDIR}/eval" --mode=eval
```

Or, for _spatially_ adaptive computation time:

``` bash
export SACT_LOGDIR='/tmp/resnet_sact_cifar'
bazel-bin/resnet_act_cifar_main --use_act=True --sact=True --tau=0.01 --train_log_dir="${SACT_LOGDIR}/train" --save_summaries_secs=300 &
bazel-bin/resnet_act_cifar_main --use_act=True --sact=True --tau=0.01 --checkpoint_dir="${SACT_LOGDIR}/train" --eval_dir="${SACT_LOGDIR}/eval" --mode=eval
```

To evaluate a pretrained ACT-Resnet model:

``` bash
python cifar_main.py --model=18 --model_type=act --tau=0.001 --checkpoint_dir='models/cifar10_resnet_18_act_0.001/train' --mode=eval --eval_dir='/tmp' --evaluate_once
```

This model is expected to achieve an accuracy of 0.9386, with the output looking
like so:

```
...
I tensorflow/core/kernels/logging_ops.cc:79] block_1/flops_std[12406060]
I tensorflow/core/kernels/logging_ops.cc:79] block_2/flops_std[1557621.2]
I tensorflow/core/kernels/logging_ops.cc:79] eval/Mean Loss[0.39572093]
I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.9386]
I tensorflow/core/kernels/logging_ops.cc:79] block_2/num_units_executed[18]
I tensorflow/core/kernels/logging_ops.cc:79] block_2/ponder_cost_std[0.25528291]
...
```

Similarly, to evaluate a pretrained ImageNet model:

``` bash
cd "${HOME}/tensorflow/models/inception"
bazel build inception/download_and_preprocess_imagenet
bazel-bin/inception/download_and_preprocess_imagenet "${HOME}/tensorflow/data/imagenet"

cd "${HOME}/sact"
bazel build :resnet_act_imagenet_eval --nocheck_visibility
bazel-bin/resnet_act_imagenet_eval --use_act=True --sact=True --evaluate_once --tau=0.001 --checkpoint_dir=models/inet_resnet_v2_101_im_224_lr_0.05_sact_0.001_res_0_ker_1_bias-3_50/train --sact_kernel_size=1
```

Note that evaluation on the full validation dataset will take some time
using only CPU tensorflow, add the arguments
`--num_examples=10 --batch_size=10`
for a quicker test.
See https://github.com/tensorflow/models/tree/master/inception/README.md for
more details on the imagenet dataset preprocessing.

For more detailed output on a few images:

``` bash
bazel build :resnet_act_imagenet_export --nocheck_visibility
bazel-bin/resnet_act_imagenet_export --num_examples=3 --sact_kernel_size=1 --sact=True --checkpoint_path=models/inet_resnet_v2_101_im_224_lr_0.05_sact_0.001_res_0_ker_1_bias-3_50/train/model.ckpt-6266164 --export_path=/tmp/maps.h5 --batch_size=1

mkdir /tmp/maps
python python/plot_ponder_maps.py --input_file=/tmp/maps.h5 --output_dir=/tmp/maps
```

DISCLAIMER
==========

This is not an official Google product.
