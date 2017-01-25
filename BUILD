# Builds files to set up the SACT codebase

package(default_visibility = [":sact"])

licenses(["notice"])  # Apache 2.0

package_group(name = "sact")

## Data providers
py_binary(
    name = "fake_cifar10",
    srcs = ["python/fake_cifar10.py"],
    deps = ["@tensorflow_models//slim:dataset_utils"],
)

py_library(
    name = "cifar_data_provider_lib",
    srcs = ["python/cifar_data_provider.py"],
    imports = ["external/tensorflow_models/slim"],
    deps = ["@tensorflow_models//slim:cifar10"],
)

py_test(
    name = "cifar_data_provider_test",
    srcs = ["python/cifar_data_provider_test.py"],
    data = [
        "testdata/cifar10/cifar10_test.tfrecord",
        "testdata/cifar10/cifar10_train.tfrecord",
    ],
    deps = [":cifar_data_provider_lib"],
)

py_binary(
    name = "fake_imagenet",
    srcs = ["python/fake_imagenet.py"],
    deps = ["@tensorflow_models//inception/inception:build_imagenet_data"],
)

py_library(
    name = "imagenet_data_provider_lib",
    srcs = ["python/imagenet_data_provider.py"],
    data = [
        "testdata/imagenet/train-00000-of-00001",
        "testdata/imagenet/validation-00000-of-00001",
    ],
    imports = ["external/tensorflow_models/slim"],
    deps = [
        "@tensorflow_models//inception/inception:image_processing",
        "@tensorflow_models//slim:imagenet",
    ],
)

py_test(
    name = "imagenet_data_provider_test",
    srcs = ["python/imagenet_data_provider_test.py"],
    deps = [":imagenet_data_provider_lib"],
)

## Common utility code
py_library(
    name = "act_lib",
    srcs = ["python/act.py"],
)

py_test(
    name = "act_test",
    srcs = ["python/act_test.py"],
    deps = [":act_lib"],
)

py_library(
    name = "flopsometer_lib",
    srcs = ["python/flopsometer.py"],
)

py_test(
    name = "flopsometer_test",
    srcs = ["python/flopsometer_test.py"],
    deps = [":flopsometer_lib"],
)

py_library(
    name = "resnet_act_utils_lib",
    srcs = ["python/resnet_act_utils.py"],
    deps = [
        ":act_lib",
        ":flopsometer_lib",
    ],
)

py_test(
    name = "resnet_act_utils_test",
    srcs = ["python/resnet_act_utils_test.py"],
    deps = [
        ":resnet_act_utils_lib",
    ],
)

## SACT-Resnet for CIFAR-10
py_library(
    name = "resnet_act_cifar_model_lib",
    srcs = ["python/resnet_act_cifar_model.py"],
    deps = [
        ":flopsometer_lib",
        ":resnet_act_utils_lib",
    ],
)

py_test(
    name = "resnet_act_cifar_model_test",
    srcs = ["python/resnet_act_cifar_model_test.py"],
    deps = [
        ":resnet_act_cifar_model_lib",
        ":resnet_act_utils_lib",
    ],
)

py_binary(
    name = "resnet_act_cifar_main",
    srcs = ["python/resnet_act_cifar_main.py"],
    deps = [
        ":cifar_data_provider_lib",
        ":resnet_act_cifar_model_lib",
        ":resnet_act_utils_lib",
    ],
)

py_test(
    name = "resnet_act_cifar_main_test",
    srcs = ["python/resnet_act_cifar_main_test.py"],
    data = [
        "testdata/cifar10/cifar10_test.tfrecord",
        "testdata/cifar10/cifar10_train.tfrecord",
    ],
    deps = [":resnet_act_cifar_main"],
)

## Imagenet Training
py_library(
    name = "resnet_act_imagenet_model_lib",
    srcs = ["python/resnet_act_imagenet_model.py"],
    deps = [
        ":act_lib",
        ":flopsometer_lib",
        ":resnet_act_utils_lib",
    ],
)

py_test(
    name = "resnet_act_imagenet_model_test",
    srcs = ["python/resnet_act_imagenet_model_test.py"],
    deps = [
        ":resnet_act_imagenet_model_lib",
        ":resnet_act_utils_lib",
    ],
)

py_binary(
    name = "resnet_act_imagenet_train",
    srcs = ["python/resnet_act_imagenet_train.py"],
    deps = [
        ":imagenet_data_provider_lib",
        ":resnet_act_imagenet_model_lib",
        ":resnet_act_utils_lib",
    ],
)

py_test(
    name = "resnet_act_imagenet_train_test",
    srcs = ["python/resnet_act_imagenet_train_test.py"],
    data = [
        "testdata/imagenet/train-00000-of-00001",
        "testdata/imagenet/validation-00000-of-00001",
    ],
    deps = [
        ":resnet_act_imagenet_train",
    ],
)

py_binary(
    name = "resnet_act_imagenet_eval",
    srcs = ["python/resnet_act_imagenet_eval.py"],
    deps = [
        ":imagenet_data_provider_lib",
        ":resnet_act_imagenet_model_lib",
        ":resnet_act_utils_lib",
    ],
)

py_binary(
    name = "resnet_act_imagenet_export",
    srcs = ["python/resnet_act_imagenet_export.py"],
    deps = [
        ":imagenet_data_provider_lib",
        ":resnet_act_imagenet_model_lib",
        ":resnet_act_utils_lib",
    ],
)
