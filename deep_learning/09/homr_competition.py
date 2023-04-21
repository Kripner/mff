#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from homr_dataset import HOMRDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--beam_width", default=5, type=int, help="CTC beam search beam width.")
parser.add_argument("--toy", default=False, action="store_true", help="If given, use a toy dataset.")
parser.add_argument("--show_images", default=True, action="store_true", help="Show augmented images.")

MAX_HEIGHT = 191


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: tf.data.Dataset) -> None:
        self.args = args

        inputs = tf.keras.layers.Input(shape=[MAX_HEIGHT, None, HOMRDataset.C], dtype=tf.float32, ragged=True)
        dense_inputs = inputs.to_tensor()
        x = dense_inputs

        for block_dim in [48, 64, 96, 128, 160]:
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(block_dim, 3, strides=2, padding="same")(x)

        column_major = tf.transpose(x, [0, 2, 1, 3])
        columns = tf.reshape(
            column_major,
            [-1, tf.shape(column_major)[1], tf.shape(column_major)[2] * tf.shape(column_major)[3]],
        )

        columns_embed = tf.keras.layers.Dense(512)(columns)

        # TODO: try tfa.rnn.LayerNormLSTMCell
        x = tf.keras.layers.LSTM(256, return_sequences=True)(columns_embed)

        logits = tf.keras.layers.Dense(1 + len(HOMRDataset.MARKS))(x)
        # TODO: check that this is not a problem
        logits = tf.RaggedTensor.from_tensor(logits)

        super().__init__(inputs=inputs, outputs=logits)

        # lr_decay = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=args.learning_rate,
        #     decay_steps=args.epochs * len(train),
        # )
        # We compile the model with the CTC loss and EditDistance metric.
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.ctc_loss,
            metrics=[HOMRDataset.EditDistanceMetric()],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        batch_losses = tf.nn.ctc_loss(
            labels=tf.cast(gold_labels, tf.int32).to_sparse(),
            logits=tf.transpose(logits.to_tensor(), [1, 0, 2]),
            label_length=tf.cast(gold_labels.row_lengths(), dtype=tf.int32),
            logit_length=tf.cast(logits.row_lengths(), tf.int32),
            blank_index=len(HOMRDataset.MARKS),
        )
        loss = tf.reduce_mean(batch_losses)
        return loss

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        decoded, _ = tf.nn.ctc_beam_search_decoder(
            inputs=tf.transpose(logits.to_tensor(), [1, 0, 2]),
            sequence_length=tf.cast(logits.row_lengths(), tf.int32),
            beam_width=self.args.beam_width,
            top_paths=1,
        )
        predictions = tf.RaggedTensor.from_sparse(decoded[0])

        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)


def train_augment_layers(image: tf.Tensor, label: tf.Tensor):
    image = tf.keras.layers.RandomTranslation(
        height_factor=0.03, width_factor=0.01, fill_mode="constant", fill_value=0, seed=args.seed
    )(image)
    image = tf.keras.layers.RandomRotation(
        0.003, fill_mode="constant", fill_value=0, seed=args.seed
    )(image)  # Might not help, because it's quite too blurry
    image = tf.image.adjust_contrast(image, 2)
    return image, label


def prepare_dataset(dataset, training, args):
    def prepare_sample(sample):
        image = tf.image.convert_image_dtype(sample["image"], tf.float32)
        width = tf.shape(image)[1]
        image = tf.image.pad_to_bounding_box(image, 0, 0, MAX_HEIGHT, width)
        return image, sample["marks"]

    dataset = dataset.map(prepare_sample)
    if training:
        dataset = dataset.shuffle(1000, seed=args.seed)
        dataset = dataset.map(train_augment_layers)
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


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

    # Load the data. The "image" is a grayscale image represented using
    # a single channel of `tf.uint8`s in [0-255] range.
    homr = HOMRDataset()

    train = prepare_dataset(homr.train, training=True, args=args)
    dev = prepare_dataset(homr.dev, training=False, args=args)
    test = prepare_dataset(homr.test, training=False, args=args)

    if args.toy:
        train = train.take(10)
        dev = dev.take(10)
        test = test.take(10)

    if args.show_images:
        summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, "images"))
        with summary_writer.as_default(step=0):
            for images, _ in train.take(1):
                images = images.to_tensor()
                images = tf.reshape(images, [args.batch_size * images.shape[1], images.shape[2], images.shape[3]])
                tf.summary.image("train/batch", tf.expand_dims(images, 0))
        summary_writer.close()

    # TODO: Create the model and train it
    model = Model(args, train)

    try:
        model.fit(
            train,
            epochs=args.epochs,
            validation_data=dev,
            callbacks=[model.tb_callback],
        )
    except KeyboardInterrupt:
        print("Training interrupted.")

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        predictions = model.predict(test)

        for sequence in predictions:
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
