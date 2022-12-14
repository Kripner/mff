#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import lzma
import os
import sys
import pathlib
import pickle
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, losses
import matplotlib.pyplot as plt

# Don't delete!!
import cart_pole_pixels_environment

import wrappers
import utils

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--local_threads", default=0, type=int,
                    help="Maximum number of threads to use when --recodex is not set.")
parser.add_argument("--epsilon", default="lin{0:0.4,4000:0.1}", type=str, help="Exploration factor.")
parser.add_argument("--gamma", default=0.997, type=float, help="Discounting factor.")
parser.add_argument("--actor_learning_rate", default=0.0001, type=float, help="Learning rate of the actor network.")
parser.add_argument("--critic_learning_rate", default=0.0001, type=float, help="Learning rate of the critic network.")
parser.add_argument("--parallel_envs_count", default=16, type=int,
                    help="How many environments to run in parallel when training.")
#parser.add_argument("--n", default=8, type=int, help="n as in n-step.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size, measured in episodes.")


utils.add_universal_arguments(parser, default_directory="pole")

# TODO: instead of epsilon-greedy, try sampling the actions proportionally to their estimated value

# TODO: during the first 30 steps, don't look and go forward

# TODO: try the DVC tool

OBSERVATION_SHAPE = (80, 80, 3)
ACTIONS_NUM = 2


class Network:
    def __init__(self, actor, critic):
        self._actor = actor
        self._critic = critic

        #utils.Visualizations.show_filters(model, layer_idx=1, max_filters=20)

    @staticmethod
    def create(args):
        return Network(Network._create_actor(args), Network._create_critic(args))

    @staticmethod
    def _create_actor(args):
        inp = layers.Input(shape=OBSERVATION_SHAPE)
        conv = layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(inp)
        #pool = layers.MaxPooling2D((2, 2))(conv)
        conv_2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(conv)
        #pool_2 = layers.MaxPooling2D((2, 2))(conv_2)
        flat = layers.Flatten()(conv_2)
        dense = layers.Dense(128, activation="sigmoid")(flat)
        output = layers.Dense(ACTIONS_NUM, activation="softmax")(dense)

        actor = models.Model(inputs=inp, outputs=output)
        actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])

        actor.summary()
        return actor

    @staticmethod
    def _create_critic(args):
        inp = layers.Input(shape=OBSERVATION_SHAPE)
        conv = layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(inp)
        #pool = layers.MaxPooling2D((2, 2))(conv)
        conv_2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(conv)
        #pool_2 = layers.MaxPooling2D((2, 2))(conv_2)
        flat = layers.Flatten()(conv_2)
        dense = layers.Dense(128, activation="relu")(flat)
        output = layers.Dense(1)(dense)

        critic = models.Model(inputs=inp, outputs=output)
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.MeanSquaredError()])

        critic.summary()
        return critic

    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: entropy regularization:
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- however, the assignment can be solved
        # quite easily without it (my reference solution does not use it).
        # In any case, `tfp.distributions.Categorical` is the suitable distribution;
        # in PyTorch, it is `torch.distributions.Categorical`.

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
        return self._actor(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        return self._critic(states)

    @tf.function
    def copy_weights_from(self, other: Network) -> None:
        for var, other_var in zip(self._actor.variables, other._actor.variables):
            var.assign(other_var)
        for var, other_var in zip(self._critic.variables, other._critic.variables):
            var.assign(other_var)

    @staticmethod
    def assert_weights_equal(lhs, rhs):
        for w1, w2 in zip(lhs._actor.get_weights(), rhs._actor.get_weights()):
            assert np.isclose(w1, w2).all()
        for w1, w2 in zip(lhs._critic.get_weights(), rhs._critic.get_weights()):
            assert np.isclose(w1, w2).all()


class ReplayBuffer:
    def __init__(self, max_size, envs_count, generator):
        assert envs_count <= max_size
        self._max_size = max_size
        self._envs_count = envs_count
        self._generator = generator

        self.o = np.zeros((self._max_size, *OBSERVATION_SHAPE), dtype=int)
        self.a = np.zeros(self._max_size, dtype=int)
        self.r = np.zeros(self._max_size, dtype=float)
        self.term = np.zeros(self._max_size, dtype=bool)
        self.trunc = np.zeros(self._max_size, dtype=bool)
        self.n = np.zeros((self._max_size, *OBSERVATION_SHAPE), dtype=int)

        self.size = 0
        self._next_idx = 0

    def add_batch(self, observations, actions, rewards, terminated, truncated, next_observations):
        assert observations.shape[0] == self._envs_count

        append_after = min(self._max_size - self._next_idx, self._envs_count)
        append_at_start = self._envs_count - append_after

        i = self._next_idx
        self.o[i:i + append_after] = observations[:append_after]
        self.a[i:i + append_after] = actions[:append_after]
        self.r[i:i + append_after] = rewards[:append_after]
        self.term[i:i + append_after] = terminated[:append_after]
        self.trunc[i:i + append_after] = truncated[:append_after]
        self.n[i:i + append_after] = next_observations[:append_after]

        if append_at_start != 0:
            self.o[:append_at_start] = observations[-append_after:]
            self.a[:append_at_start] = actions[-append_after:]
            self.r[:append_at_start] = rewards[-append_after:]
            self.term[:append_at_start] = terminated[-append_after:]
            self.trunc[:append_at_start] = truncated[-append_after:]
            self.n[:append_at_start] = next_observations[-append_after:]

        self._next_idx = (self._next_idx + self._envs_count) % self._max_size
        self.size = min(self.size + self._envs_count, self._max_size)

    def sample_uniform(self, n):
        selected = self._generator.randint(0, self.size, size=n)
        return (
            self.o[selected],
            self.a[selected],
            self.r[selected],
            self.term[selected],
            self.trunc[selected],
            self.n[selected],
        )


class DQNAlgorithm(utils.TemplateAlgorithm):
    def __init__(self, args):
        super().__init__()

        self.generator = None
        self.action_space = None

        self.network = Network.create(args)
        self.target_network = None

    def init_training(self, env, config, logger, evaluations_num):
        super().init_training(env, config, logger, evaluations_num)

        assert len(env.observation_space.shape) == 4
        self.generator = np.random.RandomState(config.seed)
        self.action_space = np.arange(env.action_space[0].n)

    def init_evaluation(self, env):
        super().init_evaluation(env)

        self.action_space = np.arange(env.action_space[0].n)

    def __getstate__(self):
        return [self.network, *super().__getstate__()]

    def __setstate__(self, state):
        self.network = state[0]
        super().__setstate__(state[1:])

    def learn_AC(self, cancellation_token: utils.CancellationToken):
        print("Training (bootstrap 1-step AC) ...")

        states = self.env.reset()[0]

        while not self.should_stop_training and not cancellation_token.is_cancelled():
            actions = self.get_stochastic_actions(states)

            # TODO: don't add experiences where truncated = True (or don't learn from them)
            # TODO: try step_async & learn while the next batch is being prepared
            next_states, rewards, terminated, truncated, _ = self.env.step(actions)
            done = np.logical_or(terminated, truncated)

            self._learn_batch(states, actions, rewards, terminated, truncated, next_states)

            states = next_states
            self._on_frame_end(rewards, done)

    def _learn_batch(self, states, actions, rewards, terminated, truncated, next_states):
        # TODO: remove
        if self.frames % 10000 == 0:
            i = self.generator.randint(states.shape[0])
            f, (ax2, ax3) = plt.subplots(1, 2)
            with np.printoptions(precision=3, suppress=True):
                print(f"a: {actions[i]}, r: {rewards[i]}, term: {terminated[i]}, trunc: {truncated[i]}")
                print(f"critic(state): {self.network.predict_values(states)[i]}")
                print(f"actor(state): {self.network.predict_actions(states)[i]}")
                print(f"critic(next_state): {self.network.predict_values(next_states)[i]}")
                print(f"actor(next_state): {self.network.predict_actions(next_states)[i]}")
            ax2.imshow(states[i])
            ax3.imshow(next_states[i])
            plt.show()

            #utils.Visualizations.show_feature_map(self.network._actor, states[0], layer_idx=1)
            #utils.Visualizations.show_feature_map(self.network._actor, states[0], layer_idx=2)

        est_next_values = np.squeeze(self.network.predict_values(next_states))
        est_returns = rewards + np.logical_not(terminated) * est_next_values
        self.network.train(states, actions, est_returns)

    def learn(self, cancellation_token: utils.CancellationToken):
        print("Training (Monte Carlo) ...")

        states = self.env.reset()[0]

        batch_states = [[] for _ in range(self.envs_count)]
        batch_actions = [[] for _ in range(self.envs_count)]
        batch_returns = [[] for _ in range(self.envs_count)]

        curr_states = [[] for _ in range(self.envs_count)]
        curr_actions = [[] for _ in range(self.envs_count)]
        curr_rewards = [[] for _ in range(self.envs_count)]
        episodes_done = np.zeros(self.envs_count)
        while not self.should_stop_training and not cancellation_token.is_cancelled():
            actions = self.get_stochastic_actions(states)

            # TODO: don't add experiences where truncated = True (or don't learn from them)
            # TODO: try step_async & learn while the next batch is being prepared
            next_states, rewards, terminated, truncated, _ = self.env.step(actions)
            done = np.logical_or(terminated, truncated)

            for i in range(self.envs_count):
                curr_states[i].append(states[i])
                curr_actions[i].append(actions[i])
                curr_rewards[i].append(rewards[i])

                if done[i]:
                    G = 0
                    if truncated[i]:
                        pass  # TODO: estimate the value of next_state[i]
                    total_reward = 0
                    curr_returns = []
                    for r in reversed(curr_rewards[i]):
                        G = G * self.config.gamma + r
                        curr_returns.append(G)
                        total_reward += r

                    if not np.isclose(total_reward, 500) or self.generator.random() > 0.3:
                        batch_states[i].extend(curr_states[i])
                        batch_actions[i].extend(curr_actions[i])
                        batch_returns[i].extend(reversed(curr_returns))
                        episodes_done[i] += 1

                    curr_states[i].clear()
                    curr_actions[i].clear()
                    curr_rewards[i].clear()

                    if episodes_done[i] == self.config.batch_size:
                        self.network.train(np.array(batch_states[i]), np.array(batch_actions[i]), np.array(batch_returns[i]))
                        batch_states[i].clear()
                        batch_actions[i].clear()
                        batch_returns[i].clear()
                        episodes_done[i] = 0

            states = next_states
            self._on_frame_end(rewards, done)

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

    def get_stochastic_actions(self, states):
        policies = self.network.predict_actions(np.array(states))
        return np.array([np.random.choice(self.action_space, p=policies[i]) for i in range(self.envs_count)])

    def get_greedy_actions(self, states):
        policies = self.network.predict_actions(np.array(states))
        return np.argmax(policies, axis=1)

    def on_evaluation(self, result: utils.ValueInInterval):
        # TODO
        #if result.lower == 500:
        #    self.should_stop_training = True
        pass

    @classmethod
    def save_to(cls, alg, filename):
        # Serialize the model.
        with lzma.open(filename, "wb") as model_file:
            pickle.dump(alg, model_file)

    @classmethod
    def load_from(cls, filename) -> DQNAlgorithm:
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
        alg.network._actor.save_weights("actor-" + filename + ".h5")
        alg.network._critic.save_weights("critic-" + filename + ".h5")

    @classmethod
    def load_from_eval_only(cls, filename, args):
        alg = DQNAlgorithm(args)
        alg.network._actor.load_weights("actor-" + filename)
        alg.network._critic.load_weights("critic-" + filename)
        return alg

    @staticmethod
    def test_save_and_load(alg: DQNAlgorithm, tmp_filename="tmp_test_save_000.model"):
        DQNAlgorithm.save_to(alg, tmp_filename)
        loaded = DQNAlgorithm.load_from(tmp_filename)

        Network.assert_weights_equal(loaded.network, alg.network)

        assert loaded.done_episodes == alg.done_episodes

        os.remove(tmp_filename)


class CartPolePixels(utils.MainTemplate):
    @classmethod
    def get_algorithm_type(cls):
        return DQNAlgorithm

    @classmethod
    def init_new_model(cls, args: argparse.Namespace):
        return DQNAlgorithm(args)

    @classmethod
    def create_logger(cls, args):
        return utils.Logger(
            args,
            tracked_vars=["ep_idx", "training_rate", "epsilon", "gamma", "train_reward", "test_reward"])

    @classmethod
    def get_vector_env(cls, args):
        vector_env = gym.vector.make("CartPolePixels-v1", args.parallel_envs_count, asynchronous=True)
        # vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
        return vector_env

    @classmethod
    def resolve_unknown_action(cls, action, simple_env, args) -> bool:
        if action == "test_saving":
            alg = cls.init_new_model(args)
            DQNAlgorithm.test_save_and_load(alg)
            return True
        return False


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # TODO: remove
#    args.model = "ac-5.model"

    print("LD_LIBRARY_PATH", os.environ.get('LD_LIBRARY_PATH', None))
    print(tf.__version__)
    print(sys.executable)
    print("is_built_with_cuda:", tf.test.is_built_with_cuda())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))
    #print("\n".join(map(str, os.environ.items())))

    # Set random seeds and the number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
        np.random.seed(args.seed)
    threads = args.threads if args.recodex else args.local_threads
    if threads != 0:
        tf.config.threading.set_inter_op_parallelism_threads(threads)
        tf.config.threading.set_intra_op_parallelism_threads(threads)

    CartPolePixels.run(env, args)
    env.close()


def entry_point():
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v1"), args.seed, args.render_each)

    main(env, args)


if __name__ == "__main__":
    entry_point()
