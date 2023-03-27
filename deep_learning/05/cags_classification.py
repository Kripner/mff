#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--finetune_epochs", default=1, type=int, help="Number of finetuning epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Note that both the "image" and the "mask" images
    # are represented using `tf.uint8`s in [0-255] range.
    cags = CAGS()

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling="avg")

    # TODO: Create the model and train it
    backbone.trainable = False

    inputs = tf.keras.layers.Input((cags.H, cags.W, cags.C))
    efnet = backbone(inputs, training=False)
    outputs = tf.keras.layers.Dense(len(cags.LABELS), activation=tf.nn.softmax)(efnet)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    def prepare_element(element):
        return element["image"], element["label"]

    train = cags.train.map(prepare_element).batch(args.batch_size)
    dev = cags.dev.map(prepare_element).batch(args.batch_size)

    tb_cbk = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
    stopping_ckb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        train,
        epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_cbk, stopping_ckb],
    )

    # FINETUNE

    backbone.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    stopping_ckb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        train,
        epochs=args.finetune_epochs,
        validation_data=dev,
        callbacks=[tb_cbk, stopping_ckb],
    )

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        def prepare_test_element(element):
            return element["image"]

        test = cags.test.map(prepare_test_element).batch(args.batch_size)
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
