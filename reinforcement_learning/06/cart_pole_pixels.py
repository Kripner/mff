#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import cart_pole_pixels_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        policy_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(env.action_space.n, activation="softmax")
        ])
        policy_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.pol_learning_rate),
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                             metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
        self._policy_model = policy_model

        value_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(1)
        ])

        value_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.val_learning_rate),
                            loss=tf.keras.losses.MeanSquaredError(),
                            metrics=[tf.keras.metrics.MeanSquaredError()])
        self._value_model = value_model

    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the policy model, using `returns - predicted_baseline` as
        #   advantage estimate
        # - train the baseline model to predict `returns`
        # TODO
        baselines = self._value_model(states)
        self._policy_model.optimizer.minimize(
            lambda: self._policy_model.compiled_loss(
                actions, self._policy_model(states, training=True), sample_weight=returns - tf.squeeze(baselines)),
            var_list=self._policy_model.trainable_variables
        )
        self._value_model.optimizer.minimize(
            lambda: self._value_model.compiled_loss(returns, self._value_model(states)),
            var_list=self._value_model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._policy_model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent

        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose an action
                action = ...
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Perform training
    raise NotImplementedError()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v1"), args.seed, args.render_each)

    main(env, args)
