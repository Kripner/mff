#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=None, type=float, help="Entropy regularization weight.")
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--actor_learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--workers", default=16, type=int, help="Number of parallel workers.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Similarly to reinforce with baseline, define two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # Use independent networks for both of them, each with
        # `args.hidden_layer_size` neurons in one ReLU hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        actor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(env.action_space.n, activation="softmax")
        ])
        actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
        self._actor = actor

        critic = tf.keras.models.Sequential([
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(1)
        ])
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.MeanSquaredError()])
        self._critic = critic

    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Train the policy network using policy gradient theorem
        # and the value network using MSE.
        #
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- however, the assignment can be solved
        # quite easily without it (my reference solution does not use it).
        # In any case, `tfp.distributions.Categorical` is the suitable distribution;
        # in PyTorch, it is `torch.distributions.Categorical`.

        # TODO: entropy regularization

        baselines = self._critic(states)
        self._actor.optimizer.minimize(
            lambda: self._actor.compiled_loss(
                actions, self._actor(states, training=True), sample_weight=returns - tf.squeeze(baselines)),
            var_list=self._actor.trainable_variables
        )
        self._critic.optimizer.minimize(
            lambda: self._critic.compiled_loss(returns, self._critic(states)),
            var_list=self._critic.trainable_variables
        )

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        return self._actor(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return estimates of value function.
        return self._critic(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            policy = network.predict_actions(np.array([state]))[0]
            # TODO: Predict the action using the greedy policy.
            action = np.argmax(policy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.make(args.env, args.workers, asynchronous=True)
    states = vector_env.reset(seed=args.seed)[0]

    action_space = np.arange(env.action_space.n)
    training = True
    frame_num = 0
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using `network.predict_actions`.
            policies = network.predict_actions(np.array(states))
            actions = np.array([np.random.choice(action_space, p=policies[i]) for i in range(args.workers)])

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = np.logical_or(terminated, truncated)

            # TODO: Compute estimates of returns by one-step bootstrapping
            est_next_values = np.squeeze(network.predict_values(next_states))
            est_returns = rewards + np.logical_not(terminated) * est_next_values

            # TODO: Train network using current states, chosen actions and estimated returns
            network.train(states, actions, est_returns)

            states = next_states
            frame_num += 1

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        #print(f"after {frame_num} frames: {np.average(returns)} (min={np.min(returns)}, max={np.max(returns)})")
        if np.average(returns) >= 475:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
