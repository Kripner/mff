#!/usr/bin/env python3
import argparse
import collections
import glob
import json
import queue
import shutil
import time
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, TypeVar, Generic, Callable, Optional
from ast import literal_eval
import os
import pickle
import threading

import numpy as np


def add_universal_arguments(parser: argparse.ArgumentParser, default_directory=None):
    parser.add_argument("-d", "--directory", type=str, default=default_directory,
                        help="Where to accumulate all results.")
    parser.add_argument("-e", "--experiment", type=str, help="Name of the current experiment.")
    parser.add_argument("-m", "--model", type=str, help="From where to load a pretrained model.")
    parser.add_argument("-l", "--load", default=False, action="store_true", help="Load a pretrained model.")
    parser.add_argument("--no_save", type=bool, default=False, help="Don't save the trained model.")
    parser.add_argument("-f", "--force", default=False, action="store_true", help="Overwrite existing files.")
    parser.add_argument("-a", "--action", default="train",
                        choices=["train", "evaluate", "test_saving", "create_eval_model"],
                        help="Only evaluate a trained model.")
    parser.add_argument("--evaluate_each", type=int, help="Period of evaluation, used when training.")
    parser.add_argument("--eval_only", type=str,
                        help="Where to save the eval-only model (only used when action = create_eval_model).")


def try_parse_float(s):
    try:
        return float(s)
    except ValueError:
        return None


class KeyboardThread(threading.Thread):
    def __init__(self, input_callback=None, name='keyboard_thread'):
        super(KeyboardThread, self).__init__(name=name)
        self.input_callback = input_callback
        self.setDaemon(True)
        self.start()

    def run(self):
        while True:
            self.input_callback(input())


class FakeObservationSpace:
    def __init__(self, shape):
        self.shape = shape


class FakeVectorEnv:
    def __init__(self, simple_env):
        assert len(simple_env.observation_space.shape) == 3
        self._env = simple_env

    @property
    def observation_space(self):
        return FakeObservationSpace((1, *self._env.observation_space.shape))

    @staticmethod
    def _add_dimension(results):
        return tuple([np.array([result]) if result is not None else None for result in results])

    def step(self, actions):
        assert actions.shape[0] == 1
        results = list(self._env.step(actions[0]))

        done = results[2] or results[3]
        # If done, we automatically reset and return the new (initial) state & info.
        if done:
            results[0], results[4] = self._env.reset()

        return FakeVectorEnv._add_dimension(results)

    def reset(self, *args, **kwargs):
        results = self._env.reset(*args, **kwargs)
        return FakeVectorEnv._add_dimension(results)


class CancellationToken:
    def __init__(self):
        self.q = queue.Queue()

    def cancel(self):
        self.q.put(True)

    def is_cancelled(self):
        return not self.q.empty()


class Logger:
    def __init__(self, args, tracked_vars):
        self.log_file = os.path.join(args.directory, args.experiment, args.experiment + ".log")
        self.tracked_vars = tracked_vars

        self.fd = None

    def begin(self):
        append = os.path.isfile(self.log_file)
        self.fd = open(self.log_file, "a")
        header = " ".join(self.tracked_vars) + "\n"
        if not append:
            self.fd.write(header)
        else:
            self.fd.write("# Appending\n")
            self.fd.write("# " + header)

    def add_entry(self, **kwargs):
        self.fd.write(
            " ".join(
                [("-" if var not in kwargs or kwargs[var] is None else str(kwargs[var])) for var in self.tracked_vars])
            + "\n")

    def end(self):
        self.fd.close()
        self.fd = None


T = TypeVar("T")


class Schedule(ABC, Generic[T]):
    @abstractmethod
    def get_value_at(self, episode_idx: int) -> T:
        pass

    @staticmethod
    def try_from_string(s):
        if s.startswith("lin"):
            return LinearSchedule.try_from_string(s)
        return None


@dataclass(frozen=True)
class LinearSchedule(Schedule[float]):
    start_ep: int
    end_ep: int
    init_value: float
    final_value: float

    def get_value_at(self, episode_idx: int) -> float:
        return np.interp(episode_idx, [self.start_ep, self.end_ep], [self.init_value, self.final_value])

    @staticmethod
    def try_from_string(s):
        s = s.removeprefix("lin")
        params = literal_eval(s)
        assert len(params) == 2
        start_ep, end_ep = min(params.keys()), max(params.keys())
        return LinearSchedule(start_ep=start_ep, end_ep=end_ep,
                              init_value=params[start_ep], final_value=params[end_ep])


@dataclass(frozen=True)
class AlternatingSchedule(Schedule[bool]):
    period: int

    def get_value_at(self, episode_idx: int) -> bool:
        return episode_idx % self.period == 0


class DynamicConfig:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._schedules = {n: v for n, v in config.items() if isinstance(v, Schedule)}

        self.on_episode_end(0)

    def add(self, name, value):
        if isinstance(value, Schedule):
            self._schedules[name] = value
        else:
            self.config[name] = value

    def is_constant(self, name):
        return name in self.config and name not in self._schedules

    def on_episode_end(self, next_episode_idx):
        for name, schedule in self._schedules.items():
            self.config[name] = schedule.get_value_at(next_episode_idx)

    def __getattr__(self, item):
        return self.config.get(item)

    @staticmethod
    def from_dict(d: dict):
        config = {}
        for n, v in d.items():
            if isinstance(v, str):
                config[n] = Schedule.try_from_string(v) or try_parse_float(v) or v
            else:
                config[n] = v
        return DynamicConfig(config)


@dataclass
class ValueInInterval:
    value: float
    lower: float
    upper: float

    def __str__(self):
        return f"(avg={self.value:.2f},min={self.lower:.2f},max={self.upper:.2f})"

    @staticmethod
    def try_from_string(s):
        if s.startswith("Int("):
            s = s.removeprefix("Int(")
            assert s[-1] == ")"
            s = s.removesuffix(")")
            params = s.split(",")
            return ValueInInterval(float(params[0]), float(params[1]), float(params[2]))
        elif s.startswith("("):
            s = s.removeprefix("(")
            assert s[-1] == ")"
            s = s.removesuffix(")")
            parts = s.replace(",", "=").split("=")
            assert len(parts) == 6
            assert parts[0] == "avg"
            assert parts[2] == "min"
            assert parts[4] == "max"
            return ValueInInterval(float(parts[1]), float(parts[3]), float(parts[5]))

        return None


class TemplateAlgorithm(ABC):
    def __init__(self):
        self.env = None
        self.config = None
        self.logger = None
        self.should_stop_training = None  # Whether the stopping condition has been met.
        self.evaluations_num = None
        self.frames = None
        self.last_fps_check = None
        self.cumulative_rewards = None
        self.envs_count = None

        self.done_episodes = 0

    def init_training(self, env, config, logger, evaluations_num):
        self.env = env
        self.envs_count = env.observation_space.shape[0]
        self.config = config
        self.logger = logger
        self.should_stop_training = False
        self.evaluations_num = evaluations_num
        self.frames = 0
        self.last_fps_check = None
        self.cumulative_rewards = np.zeros(config.parallel_envs_count)

        # self.TEST = None

    def init_evaluation(self, env):
        self.env = env
        self.envs_count = env.observation_space.shape[0]

    def __getstate__(self):
        return [self.done_episodes]

    def __setstate__(self, state):
        self.done_episodes = state[0]

    @abstractmethod
    def learn(self, cancellation_token: CancellationToken):
        pass

    @abstractmethod
    def get_actions(self, states, lagged_states, more_lagged_states, indicators):
        pass

    @classmethod
    @abstractmethod
    def load_from(cls, filename):
        pass

    @classmethod
    @abstractmethod
    def save_to(cls, alg, filename):
        pass

    @classmethod
    @abstractmethod
    def save_to_eval_only(cls, alg, filename):
        pass

    @classmethod
    @abstractmethod
    def load_from_eval_only(cls, filename, args):
        pass

    @classmethod
    def load_from_smart(cls, filename, args):
        if filename.endswith(".h5"):
            return cls.load_from_eval_only(filename, args)
        else:
            return cls.load_from(filename)

    @abstractmethod
    def evaluate_episodes(self, num: Optional[int], final_evaluation=False, on_ep_end_ckb=None) \
            -> ([float], ValueInInterval):
        pass

    @abstractmethod
    def on_evaluation(self, result: ValueInInterval):
        pass

    def _on_frame_end(self, rewards, done):
        self._update_fps()

        self.cumulative_rewards += rewards
        if done.any():
            for i in range(len(done)):
                if done[i]:
                    self._on_episode_end(self.cumulative_rewards[i])
                    self.cumulative_rewards[i] = 0

    def _on_episode_end(self, train_reward):
        if self.config.evaluate_now:
            rewards, rewards_summary = self.evaluate_episodes(self.evaluations_num)
            with np.printoptions(precision=1, suppress=True):
                print(f"episode {self.done_episodes}: {rewards_summary}  {np.array(rewards)}")
        else:
            rewards_summary = None
        if self.logger is not None:
            self.logger.add_entry(ep_idx=self.done_episodes, train_reward=train_reward, test_reward=rewards_summary,
                                  **self.config.config)

        self.done_episodes += 1
        self.config.on_episode_end(next_episode_idx=self.done_episodes)

    def _update_fps(self, frames_per_check=10000):
        self.frames += 1
        if self.last_fps_check is None:
            self.last_fps_check = time.time()
        elif self.frames % frames_per_check == 0:
            now = time.time()
            duration = now - self.last_fps_check
            print(f"{frames_per_check / duration} FPS")
            self.last_fps_check = now


class MainTemplate(ABC):
    @classmethod
    @abstractmethod
    def get_algorithm_type(cls):
        pass

    @classmethod
    @abstractmethod
    def init_new_model(cls, args):
        pass

    @classmethod
    @abstractmethod
    def create_logger(cls, args):
        pass

    @classmethod
    @abstractmethod
    def get_vector_env(cls, args):
        pass

    @classmethod
    def _get_model(cls, args):
        if args.model is None:
            args.model = os.path.join(args.directory, args.experiment, args.experiment + ".model")

        if args.load:
            print(f"Loading the model from {args.model}.")
            return cls.get_algorithm_type().load_from(args.model)
        else:
            print("Initializing new model.")
            return cls.init_new_model(args)

    @classmethod
    def run(cls, simple_env, args: argparse.Namespace):
        if args.recodex:
            # Final evaluation
            if not args.model:
                raise Exception("--model must be filled in before submitting to ReCodex.")
            algorithm = cls.get_algorithm_type().load_from_smart(args.model, args)
            algorithm.init_evaluation(FakeVectorEnv(simple_env))
            while True:
                algorithm.evaluate_episodes(num=None, final_evaluation=True)
        elif args.action == "train":
            vector_env = cls.get_vector_env(args)
            cls._train(vector_env, args)
        elif args.action == "evaluate":
            vector_env = cls.get_vector_env(args)
            cls._evaluate(vector_env, args)
        elif args.action == "create_eval_model":
            if not args.model or not args.eval_only:
                raise Exception("--model and --eval_only must be filled in for the conversion.")
            alg = cls.get_algorithm_type().load_from(args.model)
            cls.get_algorithm_type().save_to_eval_only(alg, args.eval_only)
        elif not cls.resolve_unknown_action(args.action, simple_env, args):
            raise Exception(f"Action {args.action} not implemented.")

    @classmethod
    @abstractmethod
    def resolve_unknown_action(cls, action, simple_env, args) -> bool:
        pass

    @classmethod
    def _evaluate(cls, vector_env, args, reporting_interval=20):
        if not args.load:
            print("Assuming --load.")
            args.load = True

        algorithm = cls._get_model(args)
        algorithm.init_evaluation(vector_env)
        rewards = []

        def on_ep_end(test_reward):
            nonlocal rewards
            rewards.append(test_reward)
            if len(rewards) >= reporting_interval:
                print(f"[from last {len(rewards)}] avg={np.average(rewards)},"
                      f" min={np.min(rewards)}, max={np.max(rewards)}")
                rewards = []

        while True:
            algorithm.evaluate_episodes(num=None, on_ep_end_ckb=on_ep_end)

    @classmethod
    def _prepare_directory(cls, args) -> bool:
        experiment_path = os.path.join(args.directory, args.experiment)
        if not args.load and os.path.exists(experiment_path):
            if args.force:
                print(f"Removing '{experiment_path}'.")
                shutil.rmtree(experiment_path)
            else:
                print(f"Directory '{experiment_path}' is not empty, terminating.")
                return False
        os.makedirs(experiment_path, exist_ok=True)
        return True

    @classmethod
    def _train(cls, vector_env, args):
        if not args.evaluate_each:
            print("During training, --evaluate_each must be specified.")
            return
        if not args.experiment:
            print("During training, --experiment must be specified.")
            return
        if args.no_save:
            print("Warning: the model won't be saved.")

        directory_ok = cls._prepare_directory(args)
        if not directory_ok:
            return

        algorithm = cls._get_model(args)

        logger = cls.create_logger(args)
        logger.begin()

        config = DynamicConfig.from_dict(vars(args))
        config.add("evaluate_now", AlternatingSchedule(args.evaluate_each))

        algorithm.init_training(vector_env, config, logger, evaluations_num=30)

        cancel_token = CancellationToken()

        def keyboard_input(inp):
            if inp == "stop":
                cancel_token.cancel()

        keyboard_thread = KeyboardThread(keyboard_input)

        algorithm.learn(cancellation_token=cancel_token)

        if not args.no_save:
            print(f"Saving the model to {args.model}")
            cls.get_algorithm_type().save_to(algorithm, args.model)
            cls._write_experiment_info(args)
        logger.end()
        vector_env.close_extras()

    @classmethod
    def _write_experiment_info(cls, args, blacklist=("./car_racing_environment.py", "./wrappers.py")):
        print("Writing experiment info.")

        experiment_path = os.path.join(args.directory, args.experiment)
        sources_archive = os.path.join(experiment_path, "src")
        os.makedirs(sources_archive, exist_ok=True)
        for src_file in glob.glob("./*.py"):
            if src_file in blacklist:
                continue
            shutil.copy2(src_file, sources_archive)

        with open(os.path.join(experiment_path, "info.txt"), "w") as f:
            f.write(json.dumps(vars(args), indent=4))


IndicatorInfo = collections.namedtuple("IndicatorInfo", "position positive_dir can_be_negative")


class StateProcessor:
    # will be filled in
    INDICATORS_INFO = None
    IND_MAX_LENS = [6, 5, 5, 8, 8, 25, 36]
    IND_POSITIONS = [
        (93, 12),
        (93, 17),
        (93, 20),
        (93, 21),
        (93, 26),

        (86, 48),
        (86, 72),
    ]

    IND_DIRECTIONS = [
        (-1, 0),
        (-1, 0),
        (-1, 0),
        (-1, 0),
        (-1, 0),

        (0, 1),
        (0, 1),
    ]

    IND_BIDIRECTIONAL = [False, False, False, False, False, True, True]
    IND_COUNT = len(IND_POSITIONS)

    @staticmethod
    def extract_indicators(state) -> [float]:
        values = []
        for i in range(StateProcessor.IND_COUNT):
            value = StateProcessor._get_length(state, i) / StateProcessor.IND_MAX_LENS[i]
            values.append(value)
        return values

    @staticmethod
    def extract_indicators_vector(states):
        result = np.zeros([states.shape[0], StateProcessor.IND_COUNT])
        for i in range(states.shape[0]):
            indicators = StateProcessor.extract_indicators(states[i])
            result[i, :] = indicators
        return result

    @staticmethod
    def _get_length(state, indicator_idx):
        background = (0, 0, 0)  # black background

        row, col = StateProcessor.IND_POSITIONS[indicator_idx]
        d_row, d_col = StateProcessor.IND_DIRECTIONS[indicator_idx]
        negative = StateProcessor.IND_BIDIRECTIONAL[indicator_idx] and \
                   not np.array_equal(state[row - d_row, col - d_col, :], background)
        if negative:
            d_row, d_col = -d_row, -d_col
        d = 0
        while not np.array_equal(state[row, col, :], background):
            row += d_row
            col += d_col
            d += 1
        return d if not negative else -d

    # Modifies the states inplace.
    @staticmethod
    def preprocess(states):
        states = StateProcessor._remove_indicators(states)
        # TODO: check if it's really OK to modify the state inplace
        StateProcessor._remove_car(states)
        #states = StateProcessor._down_sample(states)
        states = StateProcessor._translate_colors(states)

        # TODO: is .astype(float) necessary?
        return states.astype(float)

    @staticmethod
    def _remove_indicators(states):
        states = states[:, :84, :, :]
        return states

    @staticmethod
    def _remove_car(states):
        states[:, 66:77, 45:51, :] = np.array([102, 102, 102])

    @staticmethod
    def _down_sample(states, stride=2):
        states = states[:, ::stride, ::stride, :]
        return states

    @staticmethod
    def _translate_colors(states):
        in_colors = np.array([
            [102, 102, 102],  # grey
            [102, 204, 102],  # green
        ])
        out_colors = np.array([1, 0])

        # Calculate which in_color is most similar to each pixel.
        distance = np.linalg.norm(states[:, :, :, np.newaxis] - in_colors, ord=1, axis=4)
        r = np.argmin(distance, axis=3).astype(np.uint8)

        return out_colors[r]

    car_row, car_col = 66, 45
    car_height, car_width = 11, 6
    checks_rows = [
        [car_row - 1],
        [car_row - 1],
        [car_row + car_height],
        [car_row + car_height]
    ]
    checks_cols = [
        [car_col - 1],
        [car_col + car_width],
        [car_col - 1],
        [car_col + car_width]
    ]

    # How much is the car on the grass (vs on the road) as a value from 0 to 1.
    @staticmethod
    def get_off_road_ratios(states) -> float:
        green_dist = np.linalg.norm(
            states[:, StateProcessor.checks_rows, StateProcessor.checks_cols] - [102, 204, 102],
            ord=1, axis=3)
        ratios = np.count_nonzero(green_dist < 30, axis=1) / 4
        # TODO: find a way how to make this not necessary
        return np.transpose(ratios)[0]

    @staticmethod
    def check_upper_bounds(state):
        print([StateProcessor._get_length(state, i) for i in range(StateProcessor.IND_COUNT)])
        changed = False
        for i in range(StateProcessor.IND_COUNT):
            length = abs(StateProcessor._get_length(state, i))
            if StateProcessor.IND_MAX_LENS[i] is None or StateProcessor.IND_MAX_LENS[i] < length:
                StateProcessor.IND_MAX_LENS[i] = length
                changed = True
        if changed:
            print("max lens update", StateProcessor.IND_MAX_LENS)
