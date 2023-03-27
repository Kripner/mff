#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from cags_dataset import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--show_images", default=False, action="store_true", help="Show images in tensorboard.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for convolution kernels.")


def train_augment_tf_image(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    combined = tf.concat([image, mask], axis=-1)

    if tf.random.uniform([]) >= 0.5:
        combined = tf.image.flip_left_right(combined)
    # combined = tf.image.resize_with_crop_or_pad(combined, CAGS.H + 20, CAGS.W + 20)
    combined = tf.image.resize(combined, [tf.random.uniform([], CAGS.H, CAGS.H + 100, dtype=tf.int32),
                                          tf.random.uniform([], CAGS.W, CAGS.W + 100, dtype=tf.int32)])
    combined = tf.image.crop_to_bounding_box(
        combined, target_height=CAGS.H, target_width=CAGS.W,
        offset_height=tf.random.uniform([], maxval=tf.shape(image)[0] - CAGS.H + 1, dtype=tf.int32),
        offset_width=tf.random.uniform([], maxval=tf.shape(image)[1] - CAGS.W + 1, dtype=tf.int32),
    )
    combined = tfa.image.rotate(
        combined,
        angles=tf.random.uniform([], minval=-0.1, maxval=0.1)
    )

    image, mask = combined[..., :-1], combined[..., -1:]
    return image, mask


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

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
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)

    # Extract features of different resolution. Assuming 224x224 input images
    # (you can set this explicitly via `input_shape` of the above constructor),
    # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
            "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]]
    )

    # TODO: Create the model and train it
    backbone.trainable = False

    # PREPARE DATA

    # def prepare_element(element):
    #     return element["image"], tf.image.convert_image_dtype(element["mask"], dtype=tf.float32)
    def prepare_element(element):
        return tf.image.convert_image_dtype(element["image"], dtype=tf.float32), \
               tf.image.convert_image_dtype(element["mask"], dtype=tf.float32)

    train = cags.train \
        .map(prepare_element) \
        .shuffle(1000) \
        .batch(args.batch_size) \
        .prefetch(tf.data.AUTOTUNE)
    # train = cags.train \
    #     .map(prepare_element) \
    #     .map(train_augment_tf_image) \
    #     .shuffle(1000) \
    #     .batch(args.batch_size) \
    #     .prefetch(tf.data.AUTOTUNE)
    dev = cags.dev.map(prepare_element).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # PREPARE MODEL

    inputs = tf.keras.layers.Input((cags.H, cags.W, cags.C))
    down_path = backbone(inputs, training=False)
    x = down_path[0]
    for left_layer, filters in zip(down_path[1:], [112, 48, 32, 16]):
        from_left = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(left_layer)
        conv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding="same")(x)
        x = tf.keras.layers.Add()([from_left, conv])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same", activation=tf.nn.sigmoid)(
        x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    if args.show_images:
        summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, "images"))
        with summary_writer.as_default(step=0):
            for images, masks in train.unbatch().batch(50).take(1):
                combined = tf.concat([images, tf.repeat(masks, 3, axis=-1)], axis=0)
                combined = tf.transpose(tf.reshape(combined, [10, 10 * combined.shape[1]] + combined.shape[2:]), [0, 2, 1, 3])
                combined = tf.transpose(tf.reshape(combined, [1, 10 * combined.shape[1]] + combined.shape[2:]), [0, 2, 1, 3])
                tf.summary.image("train/batch", combined)
        summary_writer.close()

    # TRAIN
    # TODO: try using IoU as a loss
    optimizer = tf.optimizers.experimental.AdamW(
        weight_decay=args.weight_decay,
        jit_compile=False
    )
    optimizer.exclude_from_weight_decay(
        var_names=["bias", "batch_normalization", "dense"]
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy("accuracy"),
            CAGS.MaskIoUMetric(),
        ],
    )

    tb_cbk = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
    stopping_ckb = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou",
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
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy("accuracy"),
            CAGS.MaskIoUMetric(),
        ],
    )

    stopping_ckb = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou",
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

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        # def prepare_test_element(element):
        #     return element["image"]
        def prepare_test_element(element):
            return tf.image.convert_image_dtype(element["image"], dtype=tf.float32)

        test = cags.test.map(prepare_test_element).batch(args.batch_size)
        test_masks = model.predict(test)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
