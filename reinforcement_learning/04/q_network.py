#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, losses

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# TODO: use more threads
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final_at", default=400, type=int, help="Training episodes.")
#parser.add_argument("--epsilon", default=1, type=float, help="Exploration factor.")
#parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
#parser.add_argument("--epsilon_final_at", default=10000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=500, type=int, help="Target update frequency.")
parser.add_argument("--model", type=str, default="model_4.h5")
# TODO: maybe train_freq = 4


class Network:
    def __init__(self, model):
        self._model = model

    @staticmethod
    def create_model(env: wrappers.EvaluationEnv, args: argparse.Namespace):
        model = models.Sequential()
        model.add(layers.Dense(args.hidden_layer_size, activation="relu", input_shape=env.observation_space.shape))
        # TODO: consider adding 1 more layer
        model.add(layers.Dense(env.action_space.n))

        model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                      loss=losses.MeanSquaredError(),
                      metrics=[metrics.MeanSquaredError()])
        return Network(model)


    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    # Also note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, q_values: np.ndarray) -> None:
        self._model.optimizer.minimize(
            lambda: self._model.compiled_loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    @tf.function
    def copy_weights_from(self, other: Network) -> None:
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    generator = np.random.RandomState(args.seed)

    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if not args.recodex:
        # Construct the network
        network = Network.create_model(env, args)
        target_network = Network.create_model(env, args)

        # Replay memory; maxlen parameter can be passed to deque for a size limit,
        # which we however do not need in this simple task.
        replay_buffer = collections.deque()
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

        epsilon = args.epsilon
        training = True
        batch_number = 0
        total_rewards = []
        while training:
            # Perform episode
            state, done = env.reset()[0], False
            total_reward = 0
            while not done:
                # You can compute the q_values of a given state by
                #   q_values = network.predict([state])[0]
                q_values = network.predict([state])[0]
                if generator.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_values)

                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

                # Append state, action, reward, done and next_state to replay_buffer
                replay_buffer.append(Transition(state, action, reward, terminated, next_state))

                if len(replay_buffer) >= args.batch_size:
                    selected = generator.randint(len(replay_buffer), size=args.batch_size)
                    # TODO: replace this mess
                    states = np.asarray([replay_buffer[i][0] for i in selected])
                    actions = np.asarray([replay_buffer[i][1] for i in selected])
                    rewards = np.asarray([replay_buffer[i][2] for i in selected])
                    terminated = np.asarray([replay_buffer[i][3] for i in selected], dtype=bool)
                    next_states = np.asarray([replay_buffer[i][4] for i in selected])

                    q_values = network.predict(states)
                    next_q_values = target_network.predict(next_states)

                    exp_returns = rewards + args.gamma * np.logical_not(terminated) * np.max(next_q_values, axis=1)
                    q_values[np.arange(args.batch_size), actions] = exp_returns
                    network.train(states, q_values)

                    batch_number += 1
                    if batch_number % args.target_update_freq == 0:
                        target_network.copy_weights_from(network)

                state = next_state

            if args.epsilon_final_at:
                epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

            total_rewards.append(total_reward)

            if np.mean(total_rewards[-10:]) > 495:
                training = False
                if args.model is not None:
                    network._model.save(args.model)
    else:
        assert args.model is not None
        network = Network(models.load_model(args.model, compile=False))

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            q_values = network.predict([state])[0]
            action = np.argmax(q_values)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
