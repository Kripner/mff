#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=22, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
# TODO: set to 32
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--block_dims", default=[32, 64, 128], nargs="+", type=int, help="Dimensions of pre-activated blocks.")
parser.add_argument("--block_size", default=2, type=int, help="Number of CNN layers per block.")
parser.add_argument("--cnn_dropout", default=0.5, type=float, help="Dropout inside CNN blocks.")
parser.add_argument("--dense_dropout", default=0.5, type=float, help="Dropout before final dense layer.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")

parser.add_argument("--show_images", default=True, action="store_true", help="Visualize augmented objects in tensorboard.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    modelnet = ModelNet(args.modelnet)

    def augment_3d_image(image, label):
        # random shift by at most 1 voxel
        image = tf.roll(image, tf.random.uniform((3,), minval=-1, maxval=2, dtype=tf.int32), (0, 1, 2))

        # random flip
        if tf.random.uniform(()) > 0.5:
            image = tf.transpose(image, (1, 0, 2, 3))
        if tf.random.uniform(()) > 0.5:
            image = tf.transpose(image, (0, 2, 1, 3))

        return image, label

    def project_3d_image(image):
        # generate 2D projections from 3D image
        xy_projection = tf.reduce_max(image, axis=2)
        xz_projection = tf.reduce_max(image, axis=1)
        yz_projection = tf.reduce_max(image, axis=0)

        # stack projections into a single image
        vis_image = tf.concat([xy_projection, xz_projection, yz_projection], axis=1)
        vis_image = tf.expand_dims(vis_image, axis=-1)

        # normalize image values to [0, 255] for visualization
        vis_image = tf.cast(vis_image / tf.reduce_max(vis_image) * 255, tf.uint8)

        return vis_image

    def visualize_images(dataset):
        summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, "images"))
        with summary_writer.as_default(step=0):
            for images in dataset.batch(100).take(1):
                images = tf.transpose(tf.reshape(images, [10, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                images = tf.transpose(tf.reshape(images, [1, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                tf.summary.image("train/batch", images)
        summary_writer.close()

    def prepare_dataset(dataset: ModelNet.Dataset, training: bool):
        def prepare_sample(sample):
            voxels = tf.cast(sample["voxels"], tf.float32)
            labels = tf.one_hot(sample["labels"], len(modelnet.LABELS))
            return voxels, labels

        dataset = dataset.dataset
        dataset = dataset.map(prepare_sample)
        if training:
            dataset = dataset.map(augment_3d_image)
            dataset = dataset.shuffle(len(dataset), seed=args.seed)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev, test = \
        prepare_dataset(modelnet.train, training=True), \
        prepare_dataset(modelnet.dev, training=False), \
        prepare_dataset(modelnet.test, training=False)

    if args.show_images:
        visualize_images(train.unbatch().map(lambda image, _: project_3d_image(image)))

    # TODO: Create the model and train it
    inputs = tf.keras.layers.Input(shape=(modelnet.D, modelnet.H, modelnet.W, modelnet.C))
    hidden = inputs
    for cnn_dim in args.block_dims:
        for i in range(args.block_size):
            stride = 2 if i == 0 else 1

            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            hidden = tf.keras.layers.Conv3D(cnn_dim, 3, strides=stride, padding="same")(hidden)

            if i != args.block_size - 1:
                hidden = tf.keras.layers.Dropout(args.cnn_dropout)(hidden)

    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dropout(args.dense_dropout)(hidden)
    outputs = tf.keras.layers.Dense(len(modelnet.LABELS), activation="softmax")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    print(f"Training on {modelnet.train.size} samples, validating on {modelnet.dev.size} samples.")
    # TODO: cosine decay
    model.fit(
        train,
        epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback],
    )

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
