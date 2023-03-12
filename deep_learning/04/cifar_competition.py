#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from cifar10 import CIFAR10

# TODO: try residual connections
# TODO: try scaling up
# TODO: try DropBlock (keep prob = 0.9)
# TODO: try learning rate decay
# TODO: for inference, try averaging outputs with differently augmented inputs
# MKNet = ",".join([
#     "CB-32-3-1-2-same",
#     "CB-32-3-1-1-same",
#     "CB-64-3-1-2-same",
#     "CB-64-3-1-1-same",
#     "CB-128-3-1-2-same",
#     "CB-128-3-1-1-same",
#     "F",
#     "H-1000",
#     "D-0.5",
# ])
# MKNet = ",".join([
#     "CB-96-5-1-2-same", "SD-0.3",
#
#     "CB-192-3-16-2-same", "SD-0.3",
#     "CB-192-3-32-1-same", "SD-0.3",
#
#     "CB-384-3-32-2-same", "SD-0.3",
#     "CB-384-3-32-1-same", "SD-0.3",
#
#     "CB-384-3-32-2-same", "SD-0.3",
#     "CB-384-3-32-1-same", "SD-0.3",
#
#     "F",
#     "H-1000",
#     "D-0.5",
# ])
# MKNet = ",".join([
#     "CB-96-5-1-2-same", "SD-0.3",
#
#     "CB-192-3-1-2-same", "SD-0.3",
#     "CB-192-3-1-1-same", "SD-0.3",
#
#     "CB-384-3-1-2-same", "SD-0.3",
#     "CB-384-3-1-1-same", "SD-0.3",
#
#     "CB-384-3-1-2-same", "SD-0.3",
#     "CB-384-3-1-1-same", "SD-0.3",
#
#     "F",
#     "H-1000",
#     "D-0.5",
# ])
MKNet = ",".join([
    "CB-96-5-1-2-same", "SD-0.3",

    "R-192_2_[" + ",".join([
        "CB-192-3-1-2-same", "SD-0.3",
        "CB-192-3-1-1-same", "SD-0.3",
    ]) + "]",

    "R-384_2_[" + ",".join([
        "CB-384-3-1-2-same", "SD-0.3",
        "CB-384-3-1-1-same", "SD-0.3",
    ]) + "]",

    "R-384_2_[" + ",".join([
        "CB-384-3-1-2-same", "SD-0.3",
        "CB-384-3-1-1-same", "SD-0.3",
    ]) + "]",

    "F",
    "H-1000",
    "D-0.5",
])

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--arch", default=MKNet, type=str, help="CNN architecture.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for convolution kernels.")
parser.add_argument("--toy", default=False, action="store_true",
                    help="Whether to use only a toy subset of the dataset.")
parser.add_argument("--augment", default=True, action="store_true", help="Whether to augment the dataset.")
parser.add_argument("--show_images", default=True, action="store_true", help="Show augmented images.")
parser.add_argument("--preactivation", default=False, action="store_true",
                    help="Use full pre-activation on conv layers.")

parser.add_argument("--action", required=True,
                    choices=["train", "evaluate", "generate"],
                    help="What action to do.")
parser.add_argument("--models", nargs="*", type=str, help="List of models to evaluate or run.")


def prepare_datasets(cifar, args):
    train, dev, test = cifar.train, cifar.dev, cifar.test
    # TODO: try adding other channels (e.g. contours)

    if args.toy:
        for dataset in [train, dev]:
            dataset.data["images"] = dataset.data["images"][:500]
            dataset.data["labels"] = dataset.data["labels"][:500]

    # TODO: why did this work? test shouldn't contain labels
    # for dataset in [train, dev, test]:
    # Convert labels to one-hot to enable label smoothing.
    for dataset in [train, dev]:
        dataset.data["labels"] = tf.one_hot(dataset.data["labels"], len(CIFAR10.LABELS))

    train = tf.data.Dataset.from_tensor_slices((train.data["images"], train.data["labels"]))
    dev = tf.data.Dataset.from_tensor_slices((dev.data["images"], dev.data["labels"]))
    test = tf.data.Dataset.from_tensor_slices((test.data["images"], test.data["labels"]))

    train = train.shuffle(5000, seed=args.seed).map(image_to_float)
    if args.augment:
        train = train.map(train_augment_tf_image)
    train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    dev = dev.map(image_to_float).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test = test.map(image_to_float).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.show_images:
        summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, "images"))
        with summary_writer.as_default(step=0):
            for images, _ in train.unbatch().batch(100).take(1):
                images = tf.transpose(tf.reshape(images, [10, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                images = tf.transpose(tf.reshape(images, [1, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                tf.summary.image("train/batch", images)
        summary_writer.close()

    return train, dev, test


def image_to_float(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return tf.image.convert_image_dtype(image, tf.float32), label


def train_augment_tf_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform([]) >= 0.5:
        image = tf.image.flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
    image = tf.image.resize(image, [tf.random.uniform([], CIFAR10.H, CIFAR10.H + 12 + 1, dtype=tf.int32),
                                    tf.random.uniform([], CIFAR10.W, CIFAR10.W + 12 + 1, dtype=tf.int32)])
    image = tf.image.crop_to_bounding_box(
        image, target_height=CIFAR10.H, target_width=CIFAR10.W,
        offset_height=tf.random.uniform([], maxval=tf.shape(image)[0] - CIFAR10.H + 1, dtype=tf.int32),
        offset_width=tf.random.uniform([], maxval=tf.shape(image)[1] - CIFAR10.W + 1, dtype=tf.int32),
    )

    image = tf.expand_dims(image, axis=0)
    image = tfa.image.random_cutout(image, mask_size=(10, 10), constant_values=0.5)
    image = tf.squeeze(image)

    return image, label


def build_model(args):
    inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = add_layers(inputs, args.arch, args)
    outputs = tf.keras.layers.Dense(len(CIFAR10.LABELS), activation=tf.nn.softmax)(hidden)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    optimizer = tf.optimizers.experimental.AdamW(
        weight_decay=args.weight_decay,
        jit_compile=False
    )
    optimizer.exclude_from_weight_decay(
        var_names=["bias", "batch_normalization", "dense"]
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model


def train_model(model, train, dev, args):
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
    stopping_cbk = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    interrupted = False
    try:
        history = model.fit(
            train,
            validation_data=dev,
            epochs=args.epochs,
            callbacks=[tb_callback, stopping_cbk]
        )
    except KeyboardInterrupt:
        interrupted = True

    model_path = os.path.join(args.logdir, "model.h5")
    print(f"\nSaving model to {model_path}")
    os.makedirs(args.logdir, exist_ok=True)
    model.save(model_path, include_optimizer=True)

    if interrupted:
        raise KeyboardInterrupt()

    return history


def load_models(models: list[str]) -> list:
    result = []
    for model_path in models:
        print(f"Loading '{model_path}'.")
        model = tf.keras.models.load_model(model_path, compile=True)
        result.append(model)
    return result


def evaluate(models, names, dataset):
    for model, name in zip(models, names):
        print(f"{name}: ")
        model.compile(metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")])
        model.evaluate(dataset.data["images"], dataset.data["labels"])


def create_ensemble(models):
    ensemble_input = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    ensemble_output = tf.keras.layers.Average()([model(ensemble_input) for model in models])
    ensemble = tf.keras.Model(inputs=ensemble_input, outputs=ensemble_output)
    ensemble.compile(metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")])
    return ensemble


def print_gpu_debug():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num logical GPUs Available: ", len(tf.config.list_logical_devices('GPU')))
    print("Visible devices: ", tf.config.get_visible_devices())
    print("is_built_with_cuda:", tf.test.is_built_with_cuda())
    print("LD_LIBRARY_PATH", os.environ.get('LD_LIBRARY_PATH', None))


def get_logdir(args):
    args.experiment_name = "{}-{}-{}".format(
        "cifar",
        datetime.datetime.now().strftime("%d_%m-%H_%M"),
        ",".join(
            "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())
            if k not in ["arch", "debug", "threads", "toy", "show_images", "action", "models"]
        )
    )
    args.logdir = os.path.join("logs", args.experiment_name)


def generate_result_file(model, test, args):
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    test_results_path = os.path.join(args.logdir, "cifar_competition_test.txt")
    with open(test_results_path, "w", encoding="utf-8") as predictions_file:
        print(f"Writing test results to {test_results_path}")
        for probs in model.predict(test, batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()
        # tf.debugging.disable_traceback_filtering()
    print_gpu_debug()

    # Create logdir name
    get_logdir(args)

    # Load data
    cifar = CIFAR10()
    train, dev, test = prepare_datasets(cifar, args)
    print(f"Train set size: {len(train) * args.batch_size}")
    print(f"Dev set size: {len(dev) * args.batch_size}")

    if args.action == "train":
        model = build_model(args)
        history = train_model(model, train, dev, args)
    else:
        if not args.models:
            print("When evaluating or generating, you must supply a model")
            return
        models = load_models(args.models)

        if len(models) == 1:
            model = models[0]
        else:
            model = create_ensemble(models)

        if args.action == "evaluate":
            if len(models) == 1:
                evaluate(models, names=args.models, dataset=dev)
            else:
                evaluate(models + [model], names=args.models + ["ensemble"], dataset=dev)
            return

    generate_result_file(model, test, args)


# Architecture construction
# -------------------------

def add_layers(input_layer, description, args):
    x = input_layer
    layers = re.split(r",\s*(?![^\[\]]*])", description)
    for layer_descr in layers:
        x = add_layer(x, layer_descr, args)
    return x


def add_layer(input_layer, description, args):
    types = {
        "C": add_convolution,
        "CB": add_convolution_with_bn,
        "M": add_max_pool,
        "R": add_residual,
        "F": add_flatten,
        "H": add_hidden,
        "D": add_dropout,
        "SD": add_spatial_dropout,
    }

    parts = description.split("-")
    layer_type = parts[0]
    param_string = description[len(layer_type) + 1:]
    return types[layer_type](input_layer, param_string, args)


def add_convolution(input_layer, param_string, args):
    params = param_string.split("-")
    filters, kernel_size, groups, stride, padding = *map(int, params[:4]), params[4]
    return tf.keras.layers.Conv2D(filters, kernel_size, stride, padding, groups=groups,
                                  activation=tf.keras.activations.swish)(input_layer)


def add_convolution_with_bn(input_layer, param_string, args):
    params = param_string.split("-")
    filters, kernel_size, groups, stride, padding = *map(int, params[:4]), params[4]
    if args.preactivation:
        x = tf.keras.layers.BatchNormalization()(input_layer)
        x = tf.keras.activations.swish(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding, groups=groups, use_bias=False)(x)
    else:
        x = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding, groups=groups, use_bias=False)(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.swish(x)
    return x


def add_max_pool(input_layer, param_string, args):
    params = param_string.split("-")
    pool_size, stride = map(int, params)
    return tf.keras.layers.MaxPool2D(pool_size, stride)(input_layer)


def add_residual(input_layer, param_string, args):
    residual_conv = "_" in param_string
    if residual_conv:
        params = param_string.split("_")
        filters, stride, layers_descr = int(params[0]), int(params[1]), params[2][1:-1]
    else:
        layers_descr = param_string[1:-1]

    output = add_layers(input_layer, layers_descr, args)
    if residual_conv:
        residual = tf.keras.layers.Conv2D(filters, 1, stride, "same")(input_layer)
    else:
        residual = input_layer
    return tf.keras.layers.Add()([residual, output])


def add_flatten(input_layer, _, args):
    return tf.keras.layers.Flatten()(input_layer)


def add_hidden(input_layer, param_string, args):
    size = int(param_string)
    return tf.keras.layers.Dense(size, activation=tf.keras.activations.swish)(input_layer)


def add_dropout(input_layer, param_string, args):
    rate = float(param_string)
    return tf.keras.layers.Dropout(rate)(input_layer)


def add_spatial_dropout(input_layer, param_string, args):
    rate = float(param_string)
    return tf.keras.layers.SpatialDropout2D(rate)(input_layer)


if __name__ == "__main__":
    global_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(global_args)
