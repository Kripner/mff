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
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
#parser.add_argument("--entropy_regularization_decay", default=1, type=float, help="Entropy regularization decay.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=20, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=8, type=int, help="Tiles to use.")
parser.add_argument("--workers", default=16, type=int, help="Number of parallel workers.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Analogously to paac, your model should contain two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # The given states are tile encoded, so they are integral indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with with `args.tiles` ones).
        # (Or you can even use embeddings for better efficiency.)
        #
        # The actor computes `mus` and `sds`, each of shape `[batch_size, actions]`.
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and a ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `tf.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.
        actor_mus = tf.keras.models.Sequential([
            #tf.keras.layers.Embedding(env.observation_space.nvec[-1], 128, input_length=args.tiles),
            tf.keras.layers.CategoryEncoding(input_shape=(args.tiles,), num_tokens=env.observation_space.nvec[-1], output_mode="multi_hot"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            # TODO: scale tf.tanh to make it more general (even if it wasn't the case that low == -1, high == 1.
            # -> low + (high - low) * ((tanh(x) + 1) / 2)
            tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.tanh)
        ])
        actor_mus.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanSquaredError()])
        self._actor_mus = actor_mus
        actor_mus.summary()

        actor_sds = tf.keras.models.Sequential([
            #tf.keras.layers.Embedding(env.observation_space.nvec[-1], 128, input_length=args.tiles),
            tf.keras.layers.CategoryEncoding(input_shape=(args.tiles,), num_tokens=env.observation_space.nvec[-1], output_mode="multi_hot"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.nn.softplus)
        ])
        actor_sds.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanSquaredError()])
        self._actor_sds = actor_sds
        actor_sds.summary()

        critic = tf.keras.models.Sequential([
            tf.keras.layers.CategoryEncoding(input_shape=(args.tiles,), num_tokens=env.observation_space.nvec[-1], output_mode="multi_hot"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(1)
        ])
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.MeanSquaredError()])
        self._critic = critic
        critic.summary()

        self.entropy_reg = args.entropy_regularization

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Run the model on given `states` and compute
        # `sds`, `mus` and predicted values. Then create `action_distribution` using
        # `tfp.distributions.Normal` class and the computed `mus` and `sds`.
        # In PyTorch, the corresponding class is `torch.distributions.Normal`.
        #
        # TODO: Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - predicted values)`
        #   and compute its mean. Note that the gradient must not flow through
        #   the predicted values (you can use `tf.stop_gradient` if necessary).
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.

        values = self._critic(states)
        with tf.GradientTape(persistent=True) as tp:
            mus = self._actor_mus(states, training=True)
            sds = self._actor_sds(states, training=True)
            action_dist = tfp.distributions.Normal(mus, sds)

            loss = - tf.reduce_mean(tf.reduce_sum(action_dist.log_prob(actions), axis=1) * (returns - values)) \
                   - self.entropy_reg * action_dist.entropy()

        mus_grad = tp.gradient(loss, self._actor_mus.trainable_variables)
        self._actor_mus.optimizer.apply_gradients(zip(mus_grad, self._actor_mus.trainable_variables))

        sds_grad = tp.gradient(loss, self._actor_sds.trainable_variables)
        self._actor_sds.optimizer.apply_gradients(zip(sds_grad, self._actor_sds.trainable_variables))
        # self._actor_mus.optimizer.minimize(
        #     loss,
        #     var_list=self._actor_mus.trainable_variables
        # )
        # self._actor_sds.optimizer.minimize(
        #     loss,
        #     var_list=self._actor_sds.trainable_variables
        # )

        self._critic.optimizer.minimize(
            lambda: self._critic.compiled_loss(returns, self._critic(states)),
            var_list=self._critic.trainable_variables
        )

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return predicted action distributions (mus and sds).
        return self._actor_mus(states), self._actor_sds(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted state-action values.
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
            # TODO: Predict the action using the greedy policy.
            mus, _ = network.predict_actions(np.array([state]))
            action = mus[0]
            state, reward, terminated, truncated, _ = env.step(action)
            if np.isnan(rewards):
                print(state, action, reward)
            done = terminated or truncated
            rewards += reward

#        args.entropy_regularization *= args.entropy_regularization_decay
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.make("MountainCarContinuous-v0", args.workers, asynchronous=True,
                                 wrappers=lambda env: wrappers.DiscreteMountainCarWrapper(env, tiles=args.tiles))
    states = vector_env.reset(seed=args.seed)[0]

    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `network.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            mus, sds = network.predict_actions(states)
            actions = np.random.normal(mus, sds)
            actions = np.clip(actions, env.action_space.low, env.action_space.high)

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = np.logical_or(terminated, truncated)

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            est_next_values = np.squeeze(network.predict_values(next_states))
            est_returns = rewards + np.logical_not(terminated) * est_next_values

            # TODO(paac): Train network using current states, chosen actions and estimated returns
            network.train(states, actions, est_returns)

            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        if np.average(returns) > 93:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles),
        args.seed, args.render_each)

    main(env, args)
