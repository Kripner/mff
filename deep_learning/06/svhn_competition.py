#!/usr/bin/env python3
import argparse
import datetime
import itertools
import os
import re

from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

import bboxes_utils
from svhn_dataset import SVHN

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=75, type=int, help="Number of epochs.")
parser.add_argument("--finetune_epochs", default=0, type=int, help="Number of finetuning epochs.")
parser.add_argument("--seed", default=11, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--pos_threshold", default=0.5, type=float, help="Positive example IoU threshold.")
parser.add_argument("--image_size", default=224, type=float, help="Image size after resizing.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--finetune_learning_rate", default=1e-5, type=float, help="Learning rate for finetuning.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for convolutional kernels.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout.")
parser.add_argument("--alpha", default=8, type=float, help="Relative weight of the rare class.")
parser.add_argument("--gamma", default=8, type=float, help="Focal loss parameter.")
parser.add_argument("--label_smoothing", default=0.2, type=float, help="Label smoothing.")
parser.add_argument("--nms_score_threshold", default=0.4, type=float, help="Score threshold for non maximum suppression.")
parser.add_argument("--nms_iou_threshold", default=0.4, type=float, help="IoU threshold for non maximum suppression.")
parser.add_argument("--bboxes_loss_weight", default=30, type=float, help="Relative weight of the loss coming from bboxes.")
parser.add_argument("--show_images", default=True, action="store_true", help="Show debug images in Tensorboard.")
parser.add_argument("--load", type=str, help="Load model from a file.")


def roi_align(feature_maps, boxes, output_shape):
    batch_size = tf.shape(feature_maps)[0]
    channels = tf.shape(feature_maps)[-1]
    boxes_size = tf.shape(boxes)[0]

    boxes = tf.tile(tf.reshape(boxes, [-1]), multiples=[batch_size])
    boxes = tf.reshape(boxes, [batch_size * boxes_size, -1])
    box_indices = tf.repeat(tf.range(batch_size), tf.ones(batch_size, dtype=tf.int32) * boxes_size)

    result = tf.image.crop_and_resize(
        feature_maps,
        boxes,
        box_indices,
        crop_size=(output_shape[0] * 2, output_shape[1] * 2),
    )
    result = tf.nn.avg_pool(result, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", data_format="NHWC")
    result = tf.reshape(result, [batch_size, boxes_size, output_shape[0], output_shape[1], channels])
    return result


def new_multiclass_focal_loss(gamma, epsilon=1e-7):
    def multiclass_focal_loss(y_true, y_pred):
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, epsilon, 1 - epsilon))

        # tf.print(tf.shape(y_true))  # 32, 19, 11
        # tf.print(tf.shape(logits))  # 32, 19, 11
        xent_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
        )

        probs = tf.gather(probs, tf.argmax(y_true, axis=-1), axis=-1)
        focal_modulation = (1 - probs) ** gamma
        loss = focal_modulation * xent_loss

        return loss

    return multiclass_focal_loss


class ForegroundAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name, invert=False):
        super().__init__(name=name)
        self.invert = invert
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        if not self.invert:
            foreground_mask = tf.not_equal(y_true, 0)
        else:
            foreground_mask = tf.equal(y_true, 0)
        equal_mask = tf.equal(y_true, y_pred)
        self.correct.assign_add(tf.math.count_nonzero(tf.logical_and(foreground_mask, equal_mask), dtype=tf.float32))
        self.total.assign_add(tf.math.count_nonzero(foreground_mask, dtype=tf.float32))

    def result(self):
        return self.correct / self.total


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, anchors):
        self.args = args
        self.anchors = anchors

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
        backbone.trainable = False

        inputs = tf.keras.layers.Input(shape=[args.image_size, args.image_size, 3])
        backbone_outputs = backbone(inputs, training=False)

        all_features_list = []
        for backbone_output in backbone_outputs:
            reduced_output = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="same")(
                backbone_output)
            features = roi_align(reduced_output, self.anchors, output_shape=(16, 16))
            all_features_list.append(features)
        all_features = tf.keras.layers.Concatenate(axis=-1)(all_features_list)
        all_features = tf.keras.layers.Dropout(args.dropout)(all_features)

        classes_hidden = self.create_head(all_features)
        bboxes_hidden = self.create_head(all_features)

        anchor_classes = tf.keras.layers.Dense(SVHN.LABELS + 1, activation="softmax")(classes_hidden)
        anchor_bboxes = tf.keras.layers.Dense(
            units=4,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
            bias_initializer=tf.keras.initializers.Zeros(),
        )(bboxes_hidden)

        outputs = {
            "anchor_classes": anchor_classes,
            "anchor_bboxes": anchor_bboxes,
        }

        super().__init__(inputs=inputs, outputs=outputs)
        self.output_names = sorted(outputs.keys())

        optimizer = tf.optimizers.experimental.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            jit_compile=False
        )
        optimizer.exclude_from_weight_decay(
            var_names=["bias", "batch_normalization", "dense"]
        )
        self.call_compile(optimizer)

        self.backbone = backbone
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def call_compile(self, optimizer):
        self.compile(
            optimizer=optimizer,
            loss={
                "anchor_classes": new_multiclass_focal_loss(gamma=self.args.gamma),
                # "anchor_classes": tf.keras.losses.CategoricalCrossentropy(),
                "anchor_bboxes": tf.keras.losses.Huber(),
            },
            loss_weights={
                "anchor_classes": 1.0,
                "anchor_bboxes": args.bboxes_loss_weight,
            },
            metrics={
                "anchor_classes": [
                    ForegroundAccuracy("foreground_acc"),
                    ForegroundAccuracy("background_acc", invert=True),
                    tf.keras.metrics.CategoricalAccuracy("acc"),
                ],
            },
            weighted_metrics={
                "anchor_bboxes": tf.keras.metrics.MeanAbsoluteError("err"),
            },
        )

    def create_head(self, backbone_features):
        hidden = backbone_features
        for _ in range(3):
            hidden = tf.keras.layers.Conv2D(256, 3, padding="same")(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)

        hidden = tf.reshape(
            hidden,
            [tf.shape(hidden)[0], self.anchors.shape[0],
             tf.shape(hidden)[-1] * tf.shape(hidden)[-2] * tf.shape(hidden)[-3]]
        )
        hidden = tf.keras.layers.Dropout(args.dropout)(hidden)

        #hidden = tf.keras.layers.Dense(256, activation="relu")(hidden)
        #hidden = tf.keras.layers.Dropout(args.dropout)(hidden)

        return hidden


class Preprocessor:
    def __init__(self, args):
        self.args = args
        self._create_anchors()

    def _create_anchors(self):
        anchors = []
        for width in [1 / 8, 1 / 5, 1 / 3]:
            for x in np.linspace(0, 1, 20):
                right = x + width
                if right > 1:
                    break
                # top, left, bottom, right
                anchors.append([0, x, 1, right])
        abs_anchors = [[coord * self.args.image_size for coord in anchor] for anchor in anchors]
        self.anchors = tf.constant(anchors, dtype=tf.float32)
        self.abs_anchors = tf.constant(abs_anchors, dtype=tf.float32)

    def preprocess_image(self, element):
        image = element["image"]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
        return image

    def augment_image(self, image):
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.05)
        return image
        # return (
        #     image,
        #     {
        #         "classes": element[1]["classes"],
        #         "bboxes": element[1]["bboxes"],
        #     }
        # )

    def prepare_element(self, element):
        image, classes, bboxes = element["image"], element["classes"], element["bboxes"]

        size = tf.cast(tf.shape(image)[1], tf.float32)
        image = self.preprocess_image(element)
        bboxes = bboxes * self.args.image_size / size

        image = self.augment_image(image)

        anchor_classes, anchor_bboxes = tf.numpy_function(
            bboxes_utils.bboxes_training,
            [self.abs_anchors, classes, bboxes, self.args.pos_threshold],
            (tf.int32, tf.float32)
        )
        anchor_classes = tf.ensure_shape(anchor_classes, [len(self.anchors)])
        anchor_bboxes = tf.ensure_shape(anchor_bboxes, [len(self.anchors), 4])

        background_mask = tf.cast(tf.equal(anchor_classes, 0), tf.float32)
        class_weights = background_mask * (1 / self.args.alpha) + (1 - background_mask) * self.args.alpha

        bboxes_mask = tf.cast(tf.not_equal(anchor_classes, 0), tf.float32)

        anchor_classes = tf.one_hot(
            anchor_classes,
            SVHN.LABELS + 1,
            on_value=1 - self.args.label_smoothing,
            off_value=self.args.label_smoothing / SVHN.LABELS,
        )

        return (
            # input
            image,
            # targets
            {
                "anchor_classes": anchor_classes,
                "anchor_bboxes": anchor_bboxes,
            },
            # sample_weights
            {
                "anchor_classes": class_weights,
                "anchor_bboxes": bboxes_mask,
            }
        )


def print_gpu_info():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num logical GPUs Available: ", len(tf.config.list_logical_devices('GPU')))
    print("Visible devices: ", tf.config.get_visible_devices())
    print("is_built_with_cuda:", tf.test.is_built_with_cuda())
    print("LD_LIBRARY_PATH", os.environ.get('LD_LIBRARY_PATH', None))


def train_model(model, train, dev, args):
    if args.show_images:
        summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, "images"))
        with summary_writer.as_default(step=0):
            for batch in train.unbatch().batch(100).take(1):
                images = batch[0]
                images = tf.transpose(tf.reshape(images, [10, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                images = tf.transpose(tf.reshape(images, [1, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                tf.summary.image("train/batch", images)
        summary_writer.close()

    stopping_ckb = tf.keras.callbacks.EarlyStopping(
        monitor="val_anchor_classes_foreground_acc",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    try:
        logs = model.fit(
            train,
            epochs=args.epochs,
            validation_data=dev,
            callbacks=[model.tb_callback, stopping_ckb],
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    # finetune
    if args.finetune_epochs > 0:
        print("Finetuning")

        model.backbone.trainable = True
        model.call_compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.finetune_learning_rate),
        )
        
        #stopping_ckb = tf.keras.callbacks.EarlyStopping(
        #    monitor="val_anchor_classes_foreground_acc",
        #    mode="max",
        #    patience=8,
        #    restore_best_weights=True,
        #    verbose=1,
        #)
        try:
            logs = model.fit(
                train,
                epochs=args.finetune_epochs,
                validation_data=dev,
                callbacks=[model.tb_callback],
            )
        except KeyboardInterrupt:
            print("Training interrupted.")

    model_path = os.path.join(args.logdir, "model.h5")
    print(f"Saving model to {model_path}")
    model.save(model_path, include_optimizer=False)


def predict_one(sample, model, preprocessor, args):
    size = tf.cast(tf.shape(sample["image"])[1], tf.float32)
    image = preprocessor.preprocess_image(sample)

    prediction = model.predict(image, verbose=False)

    all_classes_probs = tf.squeeze(prediction["anchor_classes"], axis=0)
    all_anchor_bboxes = tf.squeeze(prediction["anchor_bboxes"], axis=0)

    all_classes = tf.argmax(all_classes_probs, axis=-1)
    all_bboxes = bboxes_utils.bboxes_from_fast_rcnn(preprocessor.abs_anchors, all_anchor_bboxes)

    all_bboxes = tf.clip_by_value(all_bboxes, 0, args.image_size)

    all_bboxes = all_bboxes / (args.image_size / size)

    output_labels = []
    output_bboxes = []
    for label in range(SVHN.LABELS):
        probs = all_classes_probs[all_classes == label + 1]
        bboxes = all_bboxes[all_classes == label + 1]
        classes = all_classes[all_classes == label + 1]

        selected_indices = tf.image.non_max_suppression(
            bboxes,
            tf.gather_nd(
                probs,
                tf.transpose([tf.range(tf.shape(probs)[0]), tf.cast(classes, tf.int32)])
            ),
            # tf.gather(probs, classes, axis=-1),
            max_output_size=2,
            score_threshold=args.nms_score_threshold,
            iou_threshold=args.nms_iou_threshold,
        )
        for i in range(selected_indices.shape[0]):
            output_labels.append(label)
            output_bboxes.append(bboxes[selected_indices[i]].numpy())

    # sort by left edge
    sort_indices = tf.argsort([bbox[1] for bbox in output_bboxes], axis=0)

    return tf.gather(output_labels, sort_indices).numpy(), tf.gather(output_bboxes, sort_indices).numpy()


def predict(dataset, model, preprocessor, args):
    for sample in tqdm(dataset):
        yield predict_one(sample, model, preprocessor, args)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    print_gpu_info()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                  for k, v in sorted(vars(args).items())
                  if k not in ["load", "threads", "show_images"]))
    ))

    # Load the data
    svhn = SVHN()

    preprocessor = Preprocessor(args)
    dev = svhn.dev.map(preprocessor.prepare_element).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.load:
        model = tf.keras.models.load_model(args.load, compile=False)
    else:
        train = svhn.train.shuffle(5000).map(preprocessor.prepare_element).batch(args.batch_size).prefetch(
            tf.data.AUTOTUNE)
        model = Model(args, preprocessor.anchors)
        train_model(model, train, dev, args)

    def log_predictions(file, dataset):
        print("Predicting ...")
        predictions = predict(dataset.batch(1), model, preprocessor, args)
        # Generate test set annotations, but in `args.logdir` to allow parallel execution.
        os.makedirs(args.logdir, exist_ok=True)
        with open(file, "w", encoding="utf-8") as predictions_file:
            # TODO: Predict the digits and their bounding boxes on the test set.
            # Assume that for a single test image we get
            # - `predicted_classes`: a 1D array with the predicted digits,
            # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
            for predicted_classes, predicted_bboxes in predictions:
                output = []
                for label, bbox in zip(predicted_classes, predicted_bboxes):
                    # abs_bbox = [coord * args.image_size for coord in list(bbox)]
                    output += [int(label)] + list(bbox)
                print(*output, file=predictions_file)
        print(f"Predictions written to {file}.")

    log_predictions(os.path.join(args.logdir, "svhn_competition.txt"), svhn.test)
    log_predictions(os.path.join(args.logdir, "svhn_competition-dev.txt"), svhn.dev)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
