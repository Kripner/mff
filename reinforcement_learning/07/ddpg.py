#!/usr/bin/env python3
import argparse
import collections
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--actor_learning_rate", default=0.0001, type=float, help="Learning rate of the actor network.")
parser.add_argument("--critic_learning_rate", default=0.001, type=float, help="Learning rate of the critic network.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--max_buffer_size", default=10_000, type=int, help="Maximum size of replay buffer.")
#TODO: train_each

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create:
        # - an actor, which starts with states and returns actions.
        #   Usually, one or two hidden layers are employed. As in the
        #   paac_continuous, to keep the actions in the required range, you
        #   should apply properly scaled `tf.tanh` activation.
        #
        # - a target actor as the copy of the actor using `tf.keras.models.clone_model`.
        #
        # - a critic, starting with given states and actions, producing predicted
        #   returns. The states and actions are usually concatenated and fed through
        #   two more hidden layers, before computing the returns with the last output layer.
        #
        # - a target critic as the copy of the critic using `tf.keras.models.clone_model`.
        self.target_tau = args.target_tau

        def scaled_tanh(x):
            l, h = env.action_space.low, env.action_space.high
            return l + (h - l) * ((tf.keras.backend.tanh(x) + 1) / 2)

        self.actor = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            tf.keras.layers.Dense(env.action_space.shape[0], activation=scaled_tanh),
        ])
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate))

        self.critic = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(env.observation_space.shape[0] + env.action_space.shape[0])),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate),
                            loss=tf.keras.losses.MeanSquaredError())

        custom_objects = {"scaled_tanh": scaled_tanh}
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.target_actor = tf.keras.models.clone_model(self.actor)
            self.target_critic = tf.keras.models.clone_model(self.critic)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Separately train:
        # - the actor using the DPG loss,
        # - the critic using MSE loss.
        #
        # Furthermore, update the target actor and critic networks by
        # exponential moving average with weight `args.target_tau`. A possible
        # way to implement it inside a `tf.function` is the following:
        #   for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        with tf.GradientTape() as tape:
            actor_actions = self.actor(states, training=True)
            states_actions = tf.concat([states, actor_actions], axis=-1)
            actor_loss = -tf.math.reduce_mean(self.critic(states_actions))
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self.critic.optimizer.minimize(
            lambda: self.critic.compiled_loss(
                returns,
                self.critic(tf.concat([states, actions], axis=-1), training=True)),
            var_list=self.critic.trainable_variables
        )

        self._moving_average_update(self.actor, self.target_actor)
        self._moving_average_update(self.critic, self.target_critic)

    def _moving_average_update(self, network, target_network):
        for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted actions by the actor.
        return self.actor(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted returns -- predict actions by the target actor
        # and evaluate them using the target critic.
        actions = self.target_actor(states)
        return self.target_critic(tf.concat([states, actions], axis=-1))


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    if args.recodex:
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=args.max_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action using the greedy policy.
            action = network.predict_actions(np.array([state]))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset()[0], False
            noise.reset()
            while not done:
                # TODO: Predict actions by calling `network.predict_actions`
                # and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                # clip the actions to the `env.action_space.{low,high}` range.
                action = network.predict_actions(np.array([state]))[0]
                action += noise.sample()
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) < args.batch_size:
                    continue
                batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))

                # TODO: Perform the training
                next_values = network.predict_values(next_states)
                #est_returns = rewards + args.gamma * (np.logical_not(done) * np.squeeze(next_values))
                est_returns = rewards + args.gamma * np.squeeze(next_values)
                network.train(states, actions, est_returns)

        # Periodic evaluation
        returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        print("Evaluation after episode {}: {:.2f}".format(env.episode, np.mean(returns)))

        if np.average(returns) > -180:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
