#!/usr/bin/env python3
import argparse
import collections
import lzma
import os
import pickle
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers
import utils

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="HalfCheetah-v4", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--parallel_envs_count", default=15, type=int, help="Number of workers to run in parallel.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--actor_learning_rate", default=0.001, type=float, help="Learning rate of the actor network.")
parser.add_argument("--critic_learning_rate", default=0.001, type=float, help="Learning rate of the critic network.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--min_buffer_size", default=500, type=int, help="Minimum size of replay buffer.")
parser.add_argument("--max_buffer_size", default=1_000_000, type=int, help="Maximum size of replay buffer.")
parser.add_argument("--d", default=2, type=int, help="How many critic updates for 1 actor update.")

# TODO: train_each?

utils.add_universal_arguments(parser, default_directory="td3_cheetah")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.target_tau = args.target_tau
        self.d = args.d
        self.training_idx = tf.Variable(0)

        def scaled_tanh(x):
            l, h = env.action_space.low, env.action_space.high
            return l + (h - l) * ((tf.keras.backend.tanh(x) + 1) / 2)

        self.actor = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", name="actor_dense_1"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", name="actor_dense_2"),
            tf.keras.layers.Dense(env.action_space.shape[0], activation=scaled_tanh, name="actor_dense_3"),
        ])
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate))

        critic = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(env.observation_space.shape[0] + env.action_space.shape[0])),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", name="critic_dense_1"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", name="critic_dense_2"),
            tf.keras.layers.Dense(1, name="critic_dense_3")
        ])
        custom_objects = {"scaled_tanh": scaled_tanh}
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.target_actor = tf.keras.models.clone_model(self.actor)
            self.target_actor.compile()

            self.critic_A = critic
            self.critic_B = tf.keras.models.clone_model(critic)
            for cr in [self.critic_A, self.critic_B]:
                cr.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate),
                           loss=tf.keras.losses.MeanSquaredError())

            self.target_critic_A = tf.keras.models.clone_model(critic)
            self.target_critic_A.compile()
            self.target_critic_B = tf.keras.models.clone_model(critic)
            self.target_critic_B.compile()

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        for critic in [self.critic_A, self.critic_B]:
            critic.optimizer.minimize(
                lambda: critic.compiled_loss(
                    returns,
                    critic(tf.concat([states, actions], axis=-1), training=True)),
                var_list=critic.trainable_variables
            )

        if self.training_idx % self.d == 0:
            with tf.GradientTape() as tape:
                actor_actions = self.actor(states, training=True)
                states_actions = tf.concat([states, actor_actions], axis=-1)
                actor_loss = -tf.math.reduce_mean(self.critic_A(states_actions))
            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            self._moving_average_update(self.actor, self.target_actor)
            self._moving_average_update(self.critic_A, self.target_critic_A)
            self._moving_average_update(self.critic_B, self.target_critic_B)

        self.training_idx.assign_add(1)

    def _moving_average_update(self, network, target_network):
        for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        return self.actor(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        actions = self.target_actor(states)
        # TODO: add noise to actions
        states_actions = tf.concat([states, actions], axis=-1)
        return tf.math.minimum(self.target_critic_A(states_actions), self.target_critic_B(states_actions))

    @staticmethod
    def assert_weights_equal(lhs, rhs):
        for net_a, net_b in zip(
                [lhs.actor, lhs.target_actor, lhs.critic_A, lhs.target_critic_A, lhs.critic_B, lhs.target_critic_B],
                [rhs.actor, rhs.target_actor, rhs.critic_A, rhs.target_critic_A, rhs.critic_B, rhs.target_critic_B]):
            for w1, w2 in zip(net_a.get_weights(), net_b.get_weights()):
                assert np.isclose(w1, w2).all()


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


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])


class TD3Algorithm(utils.TemplateAlgorithm):
    def __init__(self, simple_env, args):
        super().__init__()

        self.generator = None
        self.action_space = None
        self.replay_buffer = None
        self.noise = None

        self.actions_num = simple_env.action_space.shape[0]
        self.network = Network(simple_env, args)

    def init_training(self, env, config, logger, evaluations_num):
        super().init_training(env, config, logger, evaluations_num)

        self.generator = np.random.RandomState(config.seed)
        self.action_space = np.arange(self.actions_num)
        self.replay_buffer = collections.deque(maxlen=args.max_buffer_size)
        # self.noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
        self.noise = OrnsteinUhlenbeckNoise(env.action_space.shape, 0, args.noise_theta, args.noise_sigma)

    def init_evaluation(self, env):
        super().init_evaluation(env)

        self.action_space = np.arange(self.actions_num)

    def __getstate__(self):
        return [self.network, self.actions_num, *super().__getstate__()]

    def __setstate__(self, state):
        self.network = state[0]
        self.actions_num = state[1]
        super().__setstate__(state[2:])

    def get_greedy_actions(self, states):
        return self.network.predict_actions(states)

    def learn(self, cancellation_token: utils.CancellationToken):
        print("Training (TD3) ...")

        self.training = True
        states = self.env.reset()[0]
        while self.training:
            # TODO?
            # noise.reset()

            actions = self.get_greedy_actions(states)
            actions += self.noise.sample()
            actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            next_states, rewards, terminated, truncated, _ = self.env.step(actions)
            shaped_rewards = self._shape_rewards(states, rewards)
            done = np.logical_or(terminated, truncated)

            assert self.envs_count == states.shape[0]
            for i in range(self.envs_count):
                self.replay_buffer.append(Transition(states[i], actions[i], shaped_rewards[i], done[i], next_states[i]))

            states = next_states
            self._on_frame_end(rewards, done)

            if len(self.replay_buffer) >= args.min_buffer_size:
                self._learn_from_buffer()

            if cancellation_token.is_cancelled():
                self.training = False

    def _shape_rewards(self, states, rewards):
        inverted = states[:, 1] > 1.4  # whether the cheetah is too tilted (or upside down)
        shaped = np.logical_not(inverted) * rewards + inverted * (rewards - 5)
        return shaped

    def _learn_from_buffer(self):
        batch = np.random.randint(len(self.replay_buffer), size=args.batch_size)
        states, actions, rewards, dones, next_states = map(np.array, zip(*[self.replay_buffer[i] for i in batch]))

        next_values = self.network.predict_values(next_states)
        est_returns = rewards + args.gamma * (np.logical_not(dones) * np.squeeze(next_values))
        # est_returns = rewards + args.gamma * np.squeeze(next_values)
        self.network.train(states, actions, est_returns)

    def evaluate_episodes(self, num: Optional[int], final_evaluation=False,
                          on_ep_end_ckb=None) -> ([float], utils.ValueInInterval):
        assert num is None or num > 0

        if final_evaluation:
            states = self.env.reset(start_evaluation=True)[0]
        else:
            states = self.env.reset()[0]

        # TODO: work around this
        assert num is None or num >= self.envs_count
        started_num = self.envs_count
        results = []
        cumulative_rewards = np.zeros(self.envs_count)
        # To avoid bias, every episode that get started is also finished. Otherwise, a lot of very short episodes could
        # cause long ones to not be considered at all. Since environments are reset automatically, this mask says which
        # one are ignored, meaning they were started after the last considered episode.
        ignored = np.zeros(self.envs_count, dtype=bool)

        evaluating = True
        while evaluating:
            actions = self.get_greedy_actions(states)
            states, rewards, terminated, truncated, _ = self.env.step(actions)

            cumulative_rewards += rewards
            done = np.logical_or(terminated, truncated)

            for i in range(self.envs_count):
                if not ignored[i] and done[i]:
                    results.append(cumulative_rewards[i])
                    if on_ep_end_ckb is not None:
                        on_ep_end_ckb(cumulative_rewards[i])

                    if num:
                        if started_num == num:
                            ignored[i] = True
                            if ignored.all():
                                evaluating = False
                        else:
                            cumulative_rewards[i] = 0
                            started_num += 1

        result = utils.ValueInInterval(np.average(results), min(results), max(results))
        self.on_evaluation(result)
        return results, result

    def on_evaluation(self, result: utils.ValueInInterval):
        # TODO
        if result.lower > 8000:
            self.training = False
        pass

    @classmethod
    def save_to(cls, alg, filename: str):
        with lzma.open(filename, "wb") as model_file:
            pickle.dump(alg, model_file)

    @classmethod
    def load_from(cls, filename: str, simple_env=None) -> "TD3Algorithm":
        with lzma.open(filename, "rb") as model_file:
            return pickle.load(model_file)

    #    @classmethod
    #    def save_to_eval_only(cls, alg, filename):
    #        alg.network.save(filename + ".h5", include_optimizer=False, save_traces=False)
    #
    #    @classmethod
    #    def load_from_eval_only(cls, filename):
    #        alg = DQNAlgorithm.__new__(DQNAlgorithm)
    #        net = models.load_model(filename + ".h5")
    #        alg.network = Network(net)
    #        return alg

    @classmethod
    def save_to_eval_only(cls, alg, filename):
        if not filename.endswith(".h5"):
            filename += ".h5"
        alg.network.actor.save_weights(filename, save_format="h5")

    @classmethod
    def load_from_eval_only(cls, filename, simple_env, args):
        alg = TD3Algorithm(simple_env, args)
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        alg.network.predict_actions(np.array([simple_env.observation_space.sample()]))
        alg.network.actor.load_weights(filename)
        return alg

    @staticmethod
    def test_save_and_load(alg: "TD3Algorithm", tmp_filename="tmp_test_save_000.model"):
        TD3Algorithm.save_to(alg, tmp_filename)
        loaded = TD3Algorithm.load_from(tmp_filename)

        Network.assert_weights_equal(loaded.network, alg.network)
        assert loaded.done_episodes == alg.done_episodes

        os.remove(tmp_filename)


class WalkerTemplate(utils.MainTemplate):
    @classmethod
    def get_algorithm_type(cls):
        return TD3Algorithm

    @classmethod
    def init_new_model(cls, simple_env, args: argparse.Namespace):
        return TD3Algorithm(simple_env, args)

    @classmethod
    def create_logger(cls, args):
        return utils.Logger(
            args,
            tracked_vars=["ep_idx", "train_reward", "test_reward"])

    @classmethod
    def get_vector_env(cls, args):
        vector_env = gym.vector.make(args.env, args.parallel_envs_count, asynchronous=True)
        if args.seed:
            vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
        return vector_env

    @classmethod
    def resolve_unknown_action(cls, action, simple_env, args) -> bool:
        if action == "test_saving":
            alg = cls.init_new_model(simple_env, args)
            TD3Algorithm.test_save_and_load(alg)
            return True
        return False


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    if args.recodex:
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    print("LD_LIBRARY_PATH", os.environ.get('LD_LIBRARY_PATH', None))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # TODO: remove
    #args.model = "default3.model"

    WalkerTemplate.run(env, args)
    env.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
