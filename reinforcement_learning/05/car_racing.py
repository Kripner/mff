#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import lzma
import os
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
import car_racing_environment

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
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
# TODO: tinker with this
parser.add_argument("--epsilon", default="lin{0:0.4,4000:0.1}", type=str, help="Exploration factor.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1, type=int,
                    help="Target update frequency (measured in episodes).")
parser.add_argument("--min_buffer_size", default=1000, type=int, help="Minimum size of replay buffer to start training.")
parser.add_argument("--max_buffer_size", default=2000, type=int, help="Maximum size of replay buffer.")
parser.add_argument("--parallel_envs_count", default=16, type=int,
                    help="How many environments to run in parallel when training.")
# TODO: train_freq?

utils.add_universal_arguments(parser, default_directory="racing")

# TODO: instead of epsilon-greedy, try sampling the actions proportionally to their estimated value

# TODO: during the first 30 steps, don't look and go forward

# TODO: try the DVC tool

# TODO: extend this (soft actions + some subset of the cartesian product)
# TODO: to improve stability, maybe only use soft actions
actions_mapping = np.array([
    [0, 0, 0],  # Do Nothing
    [-0.6, 0, 0],  # Turn Left
    [0.6, 0, 0],  # Turn Right
    [0, 0.6, 0],  # Accelerate
    [0, 0.2, 0],  # Soft Accelerate
    [0, 0, 0.7],  # Break
    [0, 0, 0.2],  # Soft Break
])

ACTIONS_NUM = len(actions_mapping)

# OBSERVATION_SHAPE = (28, 32)
# OBSERVATION_SHAPE_W_CHANNELS = (28, 32, 1)
# OBSERVATION_SHAPE = (42, 48)
# OBSERVATION_SHAPE_W_CHANNELS = (*OBSERVATION_SHAPE, 2)
OBSERVATION_SHAPE = (84, 96)
OBSERVATION_SHAPE_W_CHANNELS = (*OBSERVATION_SHAPE, 3)


class RewardShaping:
    OFF_ROAD_PENALTY = 3
    SKID_PENALTY = 7
    STOP_PENALTY = 7
    SLOW_PENALTY = 0.5

    @staticmethod
    def shape_rewards(rewards, next_states_raw, next_state_indicators):
        off_road_ratios = utils.StateProcessor.get_off_road_ratios(next_states_raw)
        rewards -= RewardShaping.OFF_ROAD_PENALTY * off_road_ratios

        gyroscope = next_state_indicators[:, 6]
        rewards -= (np.abs(gyroscope) > 0.3) * (RewardShaping.SKID_PENALTY * (np.abs(gyroscope) - 0.3) / 0.7)

        speed = next_state_indicators[:, 0]
        rewards -= (speed < 0.17) * RewardShaping.SLOW_PENALTY
        rewards -= (speed < 0.1) * (RewardShaping.STOP_PENALTY - RewardShaping.SLOW_PENALTY)

        return rewards


class Network:
    def __init__(self, model):
        self._model = model

    def save(self, *args, **kwargs):
        self._model.save(*args, **kwargs)

    @staticmethod
    def create_model(args):
        states_input = layers.Input(shape=OBSERVATION_SHAPE_W_CHANNELS)

        conv = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(states_input)
        # pool = layers.MaxPooling2D((2, 2))(conv)
        conv_2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        # pool_2 = layers.MaxPooling2D((2, 2))(conv_2)
        flat = layers.Flatten()(conv_2)

        indicators_input = layers.Input(shape=utils.StateProcessor.IND_COUNT)
        merged = layers.Concatenate()([flat, indicators_input])
        dense = layers.Dense(256, activation='relu')(merged)
        output = layers.Dense(ACTIONS_NUM)(dense)

        model = models.Model(inputs=[states_input, indicators_input], outputs=output)

        model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                      loss=losses.Huber(),
                      metrics=[metrics.MeanSquaredError()])
        model.summary()

        return Network(model)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, indicators: np.ndarray, q_values: np.ndarray) -> None:
        self._model.optimizer.minimize(
            lambda: self._model.compiled_loss(q_values, self._model([states, indicators], training=True)),
            var_list=self._model.trainable_variables
        )

    @wrappers.typed_np_function(np.float32, np.float32)
    @tf.function
    def predict(self, states: np.ndarray, indicators: np.ndarray) -> np.ndarray:
        return self._model([states, indicators])

    @tf.function
    def copy_weights_from(self, other: Network) -> None:
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)

    def get_weights(self):
        return self._model.get_weights()


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])


class ReplayBuffer:
    def __init__(self, max_size, envs_count, generator):
        assert envs_count <= max_size
        self._max_size = max_size
        self._envs_count = envs_count
        self._generator = generator

        self.o = np.zeros((self._max_size, *OBSERVATION_SHAPE), dtype=int)
        self.i = np.zeros((self._max_size, utils.StateProcessor.IND_COUNT), dtype=int)
        self.a = np.zeros(self._max_size, dtype=int)
        self.r = np.zeros(self._max_size, dtype=float)
        self.term = np.zeros(self._max_size, dtype=bool)
        self.trunc = np.zeros(self._max_size, dtype=bool)
        self.n = np.zeros((self._max_size, *OBSERVATION_SHAPE), dtype=int)
        self.ni = np.zeros((self._max_size, utils.StateProcessor.IND_COUNT), dtype=int)

        self.size = 0
        self._next_idx = 0

    def add_batch(self, observations, indicators, actions, rewards, terminated, truncated, next_observations,
                  next_indicators):
        assert observations.shape[0] == self._envs_count

        append_after = min(self._max_size - self._next_idx, self._envs_count)
        append_at_start = self._envs_count - append_after

        i = self._next_idx
        self.o[i:i + append_after] = observations[:append_after]
        self.i[i:i + append_after] = indicators[:append_after]
        self.a[i:i + append_after] = actions[:append_after]
        self.r[i:i + append_after] = rewards[:append_after]
        self.term[i:i + append_after] = terminated[:append_after]
        self.trunc[i:i + append_after] = truncated[:append_after]
        self.n[i:i + append_after] = next_observations[:append_after]
        self.ni[i:i + append_after] = next_indicators[:append_after]

        if append_at_start != 0:
            self.o[:append_at_start] = observations[-append_after:]
            self.i[:append_at_start] = indicators[-append_after:]
            self.a[:append_at_start] = actions[-append_after:]
            self.r[:append_at_start] = rewards[-append_after:]
            self.term[:append_at_start] = terminated[-append_after:]
            self.trunc[:append_at_start] = truncated[-append_after:]
            self.n[:append_at_start] = next_observations[-append_after:]
            self.ni[:append_at_start] = next_indicators[-append_after:]

        self._next_idx = (self._next_idx + self._envs_count) % self._max_size
        self.size = min(self.size + self._envs_count, self._max_size)

    def sample_uniform(self, n):
        min_idx = 0 if self.size == self._max_size else 2 * self._envs_count
        selected = self._generator.randint(min_idx, self.size, size=n)
        lagged = np.mod(selected - self._envs_count, self.size)
        more_lagged = np.mod(selected - 2 * self._envs_count, self.size)
        return (
            self.o[more_lagged],
            self.i[more_lagged],
            self.o[lagged],
            self.i[lagged],
            self.o[selected],
            self.i[selected],
            self.a[selected],
            self.r[selected],
            self.term[selected],
            self.trunc[selected],
            self.n[selected],
            self.ni[selected],
        )


class DQNAlgorithm(utils.TemplateAlgorithm):
    def __init__(self, args):
        super().__init__()

        self.replay_buffer = None
        self.generator = None

        self.network = Network.create_model(args)
        self.target_network = None

    def init_training(self, env, config, logger, evaluations_num):
        super().init_training(env, config, logger, evaluations_num)

        assert len(env.observation_space.shape) == 4
        self.generator = np.random.RandomState(config.seed)
        self.replay_buffer = ReplayBuffer(max_size=config.max_buffer_size, envs_count=self.envs_count,
                                          generator=self.generator)

        self.target_network = Network.create_model(config)
        self.target_network.copy_weights_from(self.network)

    def __getstate__(self):
        return [self.network, *super().__getstate__()]

    def __setstate__(self, state):
        self.network = state[0]
        super().__setstate__(state[1:])

    def learn(self, cancellation_token: utils.CancellationToken):
        print("Training ...")

        raw_state = self.env.reset()[0]
        indicators = utils.StateProcessor.extract_indicators_vector(raw_state)
        states = utils.StateProcessor.preprocess(raw_state)

        # TODO: what about reset transitions?
        lagged_states, more_lagged_states = None, None
        while not self.should_stop_training and not cancellation_token.is_cancelled():
            if more_lagged_states is None:
                actions = np.array([0] * self.envs_count)
            else:
                actions = self.get_stochastic_actions(states, lagged_states, more_lagged_states, indicators)
            continuous_actions = self._to_continuous_actions(actions)

            # TODO: don't add experiences where truncated = True (or don't learn from them)
            # TODO: try step_async & learn while the next batch is being prepared
            next_states_raw, rewards, terminated, truncated, _ = self.env.step(continuous_actions)

            next_indicators = utils.StateProcessor.extract_indicators_vector(next_states_raw)
            next_states = utils.StateProcessor.preprocess(next_states_raw)
            done = np.logical_or(terminated, truncated)

            shaped_rewards = RewardShaping.shape_rewards(rewards, next_states_raw, next_indicators)

            self.replay_buffer.add_batch(states, indicators, actions, shaped_rewards, terminated, truncated,
                                         next_states, next_indicators)

            if self.replay_buffer.size >= self.config.batch_size:
                self._learn_from_buffer()

            states = next_states
            indicators = next_indicators

            self._on_frame_end(rewards, done)

            more_lagged_states = lagged_states
            lagged_states = states

    def _learn_from_buffer(self):
        lagged_states, lagged_indicators, more_lagged_states, more_lagged_indicators, states, indicators, \
        actions, rewards, terminated, truncated, next_states, next_indicators = \
            self.replay_buffer.sample_uniform(self.config.batch_size)

        curr = np.stack((states, lagged_states, more_lagged_states), axis=3)
        next_ = np.stack((next_states, states, lagged_states), axis=3)

        # TODO: remove
        # if self.generator.random() < 0.05:
        #    i = self.generator.randint(lagged_states.shape[0])
        #    f, (ax2, ax3) = plt.subplots(1, 2)
        #    ax2.imshow(np.stack((states[i] * 255, lagged_states[i] * 255, more_lagged_states[i] * 255), axis=2))
        #    ax3.imshow(np.stack((next_states[i] * 255, states[i] * 255, lagged_states[i] * 255), axis=2) * 255)
        #    plt.show()

        q_values = self.network.predict(curr, indicators)
        next_q_values = self.network.predict(next_, next_indicators)
        next_q_values_target = self.target_network.predict(next_, next_indicators)

        exp_returns = rewards + self.config.gamma * np.logical_not(terminated) * \
                      next_q_values_target[np.arange(self.config.batch_size), np.argmax(next_q_values, axis=1)]
        update_idx = np.arange(self.config.batch_size)
        q_values[update_idx, actions] = np.logical_not(truncated) * exp_returns + \
                                        truncated * q_values[update_idx, actions]
        self.network.train(curr, indicators, q_values)

    def evaluate_episodes(self, num: Optional[int], final_evaluation=False,
                          on_ep_end_ckb=None) -> ([float], utils.ValueInInterval):
        if final_evaluation:
            raw_states = self.env.reset(start_evaluation=True)[0]
        else:
            raw_states = self.env.reset()[0]

        indicators = utils.StateProcessor.extract_indicators_vector(raw_states)
        states = utils.StateProcessor.preprocess(raw_states)

        total_rewards = []
        cumulative_rewards = np.zeros(self.envs_count)

        lagged_states, more_lagged_states = None, None
        evaluating = True
        while evaluating:
            if more_lagged_states is None:
                actions = np.array([0] * self.envs_count)
            else:
                actions = self.get_actions(states, lagged_states, more_lagged_states, indicators)

            raw_states, rewards, terminated, truncated, _ = self.env.step(self._to_continuous_actions(actions))
            indicators = utils.StateProcessor.extract_indicators_vector(raw_states)
            states = utils.StateProcessor.preprocess(raw_states)

            cumulative_rewards += rewards
            done = np.logical_or(terminated, truncated)

            for i in range(self.envs_count):
                if done[i]:
                    total_rewards.append(cumulative_rewards[i])
                    if on_ep_end_ckb is not None:
                        on_ep_end_ckb(cumulative_rewards[i])
                    cumulative_rewards[i] = 0
                    if num is not None and len(total_rewards) >= num:
                        evaluating = False

            more_lagged_states = lagged_states
            lagged_states = states

        result = utils.ValueInInterval(np.average(total_rewards), min(total_rewards), max(total_rewards))
        self.on_evaluation(result)
        return total_rewards, result

    def _on_episode_end(self, train_reward):
        super()._on_episode_end(train_reward)

        if self.done_episodes % self.config.target_update_freq == 0:
            self.target_network.copy_weights_from(self.network)

    def get_stochastic_actions(self, states, lagged_states, more_lagged_states, indicators):
        actions = self.get_actions(states, lagged_states, more_lagged_states, indicators)
        greedy = np.random.uniform(size=self.envs_count) > self.config.epsilon
        return greedy * actions + (1 - greedy) * self.generator.randint(0, ACTIONS_NUM, size=self.envs_count)

    def get_actions(self, states, lagged_states, more_lagged_states, indicators):
        q_values = self.network.predict(np.stack((states, lagged_states, more_lagged_states), axis=3), indicators)
        return np.argmax(q_values, axis=1)

    @staticmethod
    def _to_continuous_actions(actions):
        return actions_mapping[actions]

    def on_evaluation(self, result: utils.ValueInInterval):
        # TODO
        if result.value > 850:
            self.should_stop_training = True

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
        alg.network._model.save_weights(filename + ".h5")

    @classmethod
    def load_from_eval_only(cls, filename, args):
        alg = DQNAlgorithm(args)
        alg.network._model.load_weights(filename)
        return alg

    @staticmethod
    def test_save_and_load(alg: DQNAlgorithm, tmp_filename="tmp_test_save_000.model"):
        DQNAlgorithm.save_to(alg, tmp_filename)
        loaded = DQNAlgorithm.load_from(tmp_filename)

        # assert loaded.env is None
        # assert loaded.config is None
        # assert loaded.logger is None
        # assert loaded.should_stop_training is None
        # assert loaded.evaluations_num is None

        for w1, w2 in zip(loaded.network.get_weights(), alg.network.get_weights()):
            print(w1.shape)
            assert np.isclose(w1, w2).all()
            # tf.debugging.assert_equal(w1, w2)

        assert loaded.done_episodes == alg.done_episodes

        os.remove(tmp_filename)


class CarRacing(utils.MainTemplate):
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
        vector_env = gym.vector.make(
            "CarRacingFS{}-v2".format(args.frame_skip), args.parallel_envs_count, asynchronous=True)
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
#    args.model = "model.h5"
#    args.frame_skip = 1

    print("LD_LIBRARY_PATH", os.environ.get('LD_LIBRARY_PATH', None))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Set random seeds and the number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
        np.random.seed(args.seed)
    threads = args.threads if args.recodex else args.local_threads
    if threads != 0:
        tf.config.threading.set_inter_op_parallelism_threads(threads)
        tf.config.threading.set_intra_op_parallelism_threads(threads)

    CarRacing.run(env, args)
    env.close()


def entry_point():
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        gym.make("CarRacingFS{}-v2".format(args.frame_skip)), args.seed, args.render_each,
        evaluate_for=15, report_each=1)

    main(env, args)


if __name__ == "__main__":
    entry_point()
