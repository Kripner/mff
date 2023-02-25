#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import re
import itertools
import copy

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/local/lib/cuda:/usr/local/cuda-11.2/lib64")
os.environ.setdefault("LIBRARY_PATH", os.environ["LD_LIBRARY_PATH"])

import h5py
import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `windows`.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=75, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=10, type=int, help="Window size to use.")
parser.add_argument("--embed_dim", default=10, type=int, help="Dimension of the character embedding.")
parser.add_argument("--hidden_layers", default=[2000, 2000], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--toy", default=False, action="store_true",
                    help="Whether to use only a toy subset of the dataset.")
parser.add_argument("--action", required=True, choices=["grid_search", "list_search", "evaluate", "generate"],
                    help="What action to do.")
parser.add_argument("--models", nargs="*", type=str, help="List of models to evaluate or run.")

args = parser.parse_args([] if "__file__" not in globals() else None)

PARAM_OPTIONS = {
    "dropout": [0, 0.2],
    "label_smoothing": [0.1, 0.25],
    "window": [10, 15],
    "hidden_layers": [[2500], [500, 500], [2000, 2000], [4000, 4000], [500, 500, 500]],
}

OPTIONS_GRID = itertools.product(*[[(name, value) for value in values] for name, values in PARAM_OPTIONS.items()])

OPTIONS_LIST = [
    [("dropout", 0.0), ("label_smoothing", 0.1), ("window", 10), ("hidden_layers", [2000, 2000]), ("seed", 42)],
    [("dropout", 0.0), ("label_smoothing", 0.1), ("window", 10), ("hidden_layers", [2000, 2000]), ("seed", 0)],
    [("dropout", 0.0), ("label_smoothing", 0.1), ("window", 10), ("hidden_layers", [2000, 2000]), ("seed", 1)],

    [("dropout", 0.1), ("label_smoothing", 0.1), ("window", 15), ("hidden_layers", [4000, 4000]), ("seed", 0)],
    [("dropout", 0.3), ("label_smoothing", 0.1), ("window", 15), ("hidden_layers", [4000, 4000]), ("seed", 1)],

    [("dropout", 0.1), ("label_smoothing", 0.1), ("window", 12), ("hidden_layers", [4000, 4000]), ("seed", 0)],
]


def prepare_datasets(args):
    uppercase_data = UppercaseData(args.window, args.alphabet_size)
    train, dev, test = uppercase_data.train, uppercase_data.dev, uppercase_data.test

    # Convert labels to one-hot to enable label smoothing.
    for dataset in [train, dev, test]:
        dataset.data["labels"] = tf.one_hot(dataset.data["labels"], 2)

    if args.toy:
        for dataset in [train, dev]:
            dataset.data["windows"] = dataset.data["windows"][:500]
            dataset.data["labels"] = dataset.data["labels"][:500]

    return train, dev, test


def build_model(args):
    inp = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    emb = tf.keras.layers.Embedding(input_dim=args.alphabet_size, output_dim=args.embed_dim)(inp)
    flat = tf.keras.layers.Flatten()(emb)
    hid = flat
    for hidden_size in args.hidden_layers:
        hid = tf.keras.layers.Dense(hidden_size, activation=None, use_bias=False)(hid)
        hid = tf.keras.layers.BatchNormalization()(hid),
        hid = tf.keras.layers.ReLU()(hid),
        # TODO: why does ReLU add spurious dimension??
        hid = tf.squeeze(hid, axis=0)

        hid = tf.keras.layers.Dropout(args.dropout)(hid)
        # TODO: why does Dropout add spurious dimension??
        hid = tf.squeeze(hid, axis=0)
    out = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(hid)

    model = tf.keras.Model(inputs=inp, outputs=out)

    optimizer = tf.keras.optimizers.experimental.Adam(jit_compile=False)
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")]
    )

    return model


def train_model(model, train, dev, args):
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
    stopping_cbk = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train.data["windows"], train.data["labels"],
        validation_data=(dev.data["windows"], dev.data["labels"]),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[tb_callback, stopping_cbk]
    )
    return history


def get_logdir(args):
    args.experiment_name = ",".join(
        ("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    args.logdir = os.path.join("logs", "{}-{}".format("uppercase_prod", args.experiment_name))


def param_search(args, configurations):
    original_args = args
    for config in configurations:
        args = copy.deepcopy(original_args)
        for param, value in config:
            setattr(args, param, value)

        print("Experiment: ", args)
        get_logdir(args)
        if os.path.isdir(args.logdir):
            print("already done")
            continue

        tf.keras.utils.set_random_seed(args.seed)
        train, dev, test = prepare_datasets(args)
        model = build_model(args)
        history = train_model(model, train, dev, args)
        model.save(os.path.join("models", args.experiment_name + ".h5"), include_optimizer=False)


def load_models(models: list[str]) -> list:
    result = []
    for model_path in models:
        print(f"Loading '{model_path}'.")
        model = tf.keras.models.load_model(model_path, compile=False)
        result.append(model)
    return result


def correct_saved_model(model_path, squeeze_layer_indices, new_inbounds):
    with h5py.File(model_path, 'a') as model_file:
        config = json.loads(model_file.attrs["model_config"])
        layers = config["config"]["layers"]

        for squeeze_layer_idx, new_inbound in zip(squeeze_layer_indices, new_inbounds):
            squeeze_name = "tf.compat.v1.squeeze_" + str(squeeze_layer_idx)

            def should_remove(layer_name):
                return layer_name == squeeze_name

            layers = [layer for layer in layers if not should_remove(layer["config"]["name"])]

            for layer in layers:
                if len(layer["inbound_nodes"]) > 0:
                    inbound_nodes = layer["inbound_nodes"][0][0] if type(layer["inbound_nodes"][0][0]) == list else \
                    layer["inbound_nodes"][0]
                    if inbound_nodes[0] == squeeze_name:
                        inbound_nodes[0] = new_inbound

            config["config"]["layers"] = layers
            model_file.attrs["model_config"] = json.dumps(config)

            # del model_file["model_weights"][squeeze_name]


def correct_saved_models():
    if True:
        correct_saved_model(
            "models/as=75,bs=64,d=False,d=0.0,ed=10,e=30,gs=False,hl=[2000, 2000],ls=0.1,ls=True,s=0,t=0,t=False,w=10.h5",
            [5, 7],
            ["dropout_2", "dropout_3"]
        )
        correct_saved_model(
            "models/as=75,bs=64,d=False,d=0.0,ed=10,e=30,gs=False,hl=[2000, 2000],ls=0.1,ls=True,s=1,t=0,t=False,w=10.h5",
            [9, 11],
            ["dropout_4", "dropout_5"]
        )
        correct_saved_model(
            "models/as=75,bs=64,d=False,d=0.0,ed=10,e=30,gs=False,hl=[2000, 2000],ls=0.1,ls=True,s=42,t=0,t=False,w=10.h5",
            [1, 3],
            ["dropout", "dropout_1"]
        )


def print_model_config(model_path: str):
    with h5py.File(model_path, 'r') as model_file:
        weights: h5py.Group = model_file["model_weights"]
        print(weights.keys())
        print(model_file.attrs.get('model_config'))
        # raise Exception("Stopping")


def evaluate(models, dataset):
    for model in models:
        model.compile(metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")])
        model.evaluate(dataset.data["windows"], dataset.data["labels"])


def create_ensemble(models):
    ensemble_input = tf.keras.layers.Input(shape=[2 * args.window + 1])
    ensemble_output = tf.keras.layers.Average()([model(ensemble_input) for model in models])
    ensemble = tf.keras.Model(inputs=ensemble_input, outputs=ensemble_output)
    ensemble.compile(metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")])
    return ensemble


def uppercase(text, predictions):
    uppercased = ""
    for i in range(len(text)):
        if predictions[i][0] >= predictions[i][1]:
            uppercased += text[i].lower()
        else:
            uppercased += text[i].upper()
    return uppercased


def generate_capitalization(model, test):
    with open("uppercase_test_2.txt", "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test.data["windows"])
        uppercased = uppercase(test.text, predictions)
        predictions_file.write(uppercased)


def print_gpu_debug():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num logical GPUs Available: ", len(tf.config.list_logical_devices('GPU')))
    print("Visible devices: ", tf.config.get_visible_devices())
    print("is_built_with_cuda:", tf.test.is_built_with_cuda())
    print("LD_LIBRARY_PATH", os.environ.get('LD_LIBRARY_PATH', None))


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()
    print_gpu_debug()

    if args.action == "grid_search":
        print("Performing grid search.")
        param_search(args, OPTIONS_GRID)
    elif args.action == "list_search":
        print("Performing list search.")
        param_search(args, OPTIONS_LIST)
    elif args.action in ["evaluate", "generate"]:
        if not args.models:
            print("Specify --models.")
            return

        for model in args.models:
            print_model_config(model)
        correct_saved_models()

        models = load_models(args.models)
        _, dev, test = prepare_datasets(args)
        if args.action == "evaluate":
            evaluate(models, dataset=dev)
        else:
            if len(models) == 1:
                model = models[0]
            else:
                model = create_ensemble(models)
                model.summary()
                model(tf.zeros([1, 2 * args.window + 1]))
                #model.evaluate(dev.data["windows"], dev.data["labels"])
            generate_capitalization(model, test)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
