#!/usr/bin/env python3
import argparse
import collections
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, TypeVar, Generic, Callable
from ast import literal_eval
import os
import pickle
import threading

import numpy as np


def add_universal_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--evaluate_each", type=int, help="Period of evaluation.")
    parser.add_argument("-f", "--force", default=False, action="store_true", help="Overwrite existing files.")
    parser.add_argument("-s", "--save", type=str, help="Where to store the trained model.")
    parser.add_argument("--no_save", type=bool, default=False, help="Don't save the trained model.")
    parser.add_argument("-m", "--model", type=str, help="From where to load a pretrained model.")
    parser.add_argument("-e", "--evaluate", default=False, action="store_true", help="Only evaluate a trained model.")
    parser.add_argument("--pretrain", default=False, action="store_true",
                        help="Do pretraining using expert trajectories instead of self-learning.")
    parser.add_argument("--pretrain_mc", default=False, action="store_true",
                        help="Do pretraining using expert trajectories instead of self-learning; use Monte Carlo approach.")
    parser.add_argument("--mc", default=False, action="store_true")
    parser.add_argument("--sarsa", default=False, action="store_true")


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


class Logger:
    def __init__(self, log_directory, experiment_id, tracked_vars):
        self.log_directory = log_directory
        self.experiment_id = experiment_id
        self.tracked_vars = tracked_vars

        self.fd = None

    @property
    def filename(self):
        return os.path.join(self.log_directory, self.experiment_id + ".log")

    def begin(self, append=False, force=False):
        assert not (append and force)
        if not append and os.path.isfile(self.filename):
            if force:
                print(f"Overwriting existing file {self.filename}.")
                os.remove(self.filename)
            else:
                raise Exception(f"File {self.filename} already exists.")
        os.makedirs(self.log_directory, exist_ok=True)
        self.fd = open(self.filename, "w" if not append else "a")
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
        #progress = np.clip((episode_idx - self.start_ep) / (self.end_ep - self.start_ep + 1), 0, 1)
        #return self.init_value + progress * (self.final_value - self.init_value)

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
        return f"Int({self.value},{self.lower},{self.upper})"

    @staticmethod
    def try_from_string(s):
        if not s.startswith("Int("):
            return None
        s = s.removeprefix("Int(")
        assert s[-1] == ")"
        s = s.removesuffix(")")
        params = s.split(",")
        return ValueInInterval(float(params[0]), float(params[1]), float(params[2]))


# TODO: Split this into Model and Algorithm (algorithm = q learning, expert pretraining via monte carlo, evaluating)
class Algorithm(ABC):
    def __init__(self, evaluations_num=10):
        self.env = None
        self.config = None
        self.logger = None
        self.stopping_cond = None
        self.should_stop = None  # Whether the stopping condition has been met.

        self.evaluations_num = evaluations_num
        self.done_episodes = 0

    def init_training(self, env, config, logger, stopping_cond: Callable[[ValueInInterval], bool]):
        self.env = env
        self.config = config
        self.logger = logger
        self.stopping_cond = stopping_cond
        self.should_stop = False

    def init_evaluation(self, env):
        self.env = env

    def __getstate__(self):
        return [self.done_episodes, self.evaluations_num]

    def __setstate__(self, state):
        self.done_episodes, self.evaluations_num = state

    @abstractmethod
    def perform_episode(self):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @classmethod
    @abstractmethod
    def load_from(cls, file):
        pass

    @classmethod
    @abstractmethod
    def save_to(cls, alg, file):
        pass

    @classmethod
    def load_from_path(cls, path):
        with open(path, "rb") as f:
            return cls.load_from(f)

    @classmethod
    def save_to_path(cls, alg, path):
        with open(path, "wb") as f:
            cls.save_to(alg, f)

    def _on_episode_end(self, train_reward):
        test_reward = self.evaluate_episodes(self.evaluations_num) if self.config.evaluate_now else None
        if self.logger is not None:
            self.logger.add_entry(ep_idx=self.done_episodes, train_reward=train_reward, test_reward=test_reward,
                                  **self.config.config)

        self.done_episodes += 1
        self.config.on_episode_end(next_episode_idx=self.done_episodes)

    def evaluate_episode(self, final_evaluation=False) -> float:
        state, done = self.env.reset(start_evaluation=final_evaluation)[0], False
        total_reward = 0
        while not done:
            action = self.get_action(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward

    def evaluate_episodes(self, num: int) -> ValueInInterval:
        rewards = [self.evaluate_episode() for _ in range(num)]
        vii = ValueInInterval(np.average(rewards), min(rewards), max(rewards))
        self.should_stop = self.stopping_cond(vii)
        return vii


class QLearning(Algorithm):
    def __init__(self, env, double_q=False, init_optimistic=0, evaluations_num=10):
        super().__init__(evaluations_num)

        n_actions = env.action_space.n
        n_states = env.observation_space.n

        def create_q():
            if init_optimistic == 0:
                return np.zeros([n_states, n_actions])
            else:
                return np.ones([n_states, n_actions]) * init_optimistic

        self.double_q = double_q
        if not double_q:
            self.q = create_q()
        else:
            self.q1 = create_q()
            self.q2 = create_q()

    def __getstate__(self):
        first_q = self.q1 if self.double_q else self.q
        second_q = self.q2 if self.double_q else None
        return [self.double_q, first_q, second_q, *super().__getstate__()]

    def __setstate__(self, state):
        self.double_q, first_q, second_q, *super_state = state
        if self.double_q:
            self.q1, self.q2 = first_q, second_q
        else:
            self.q = first_q
            assert second_q is None
        super().__setstate__(super_state)

    def perform_episode(self):
        state, done = self.env.reset()[0], False
        total_reward = 0
        while not done:
            action = self.choose_next_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

            self.learn_from_step(state, action, reward, next_state)
            state = next_state

        self._on_episode_end(total_reward)

    def perform_episode_mc(self):
        state, done = self.env.reset()[0], False
        total_reward = 0
        trajectory = []
        while not done:
            action = self.choose_next_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            trajectory.append([state, action, reward])
            state = next_state
            done = terminated or truncated
            total_reward += reward
        trajectory.append((state, None, None))
        self.learn_from_episode(trajectory)

        self._on_episode_end(total_reward)

    def choose_next_action(self, state):
        # Perform an action.
        if np.random.uniform() < self.config.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_action(state)

    def perform_episode_sarsa(self, n=1):
        assert (self.config.is_constant("gamma"))
        next_state, done = self.env.reset()[0], False

        # Generate episode and update Q using the given TD method
        next_action = self.choose_next_action(next_state)
        buffer = collections.deque()
        g = 0
        gamma_n = pow(self.config.gamma, n)
        while not done:
            action, state = next_action, next_state
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if not done:
                next_action = self.choose_next_action(next_state)

            buffer.appendleft((state, action, reward))
            g += reward * (gamma_n / self.config.gamma)
            if len(buffer) == n:
                past_state, past_action, past_reward = buffer.pop()
                val_estimate = g
                if not done:
                    curr_val_estimate = self.q[next_state, next_action]
                    val_estimate += gamma_n * curr_val_estimate
                self.q[past_state, past_action] += self.config.alpha * (val_estimate - self.q[past_state, past_action])

                g = (g - past_reward) / self.config.gamma
            else:
                g = g / self.config.gamma

        while len(buffer) > 0:
            past_state, past_action, past_reward = buffer.pop()
            # TODO: this is where different modes will differ, this is "sarsa"
            self.q[past_state, past_action] += self.config.alpha * (g - self.q[past_state, past_action])
            g = (g - past_reward) / self.config.gamma

    def learn_from_step(self, state, action, reward, next_state):
        # Update the action-value estimates
        q1, q2 = (self.q, self.q) if not self.double_q else (self.q1, self.q2)
        next_action = np.argmax(q1[next_state, :])
        value_est = reward + self.config.gamma * q2[next_state, next_action]
        q1[state, action] += self.config.alpha * (value_est - q1[state, action])

        if self.double_q and np.random.uniform() < 0.5:
            self.q1, self.q2 = self.q2, self.q1

    def learn_from_episode(self, trajectory):
        G = 0
        for i in range(len(trajectory) - 2, -1, -1):
            state, action, reward, next_state = *trajectory[i], trajectory[i + 1][0]
            G = G * self.config.gamma + reward

            # Update the action-value estimates
            q1, q2 = (self.q, self.q) if not self.double_q else (self.q1, self.q2)
            q1[state, action] += self.config.alpha * (G - q1[state, action])

            if self.double_q and np.random.uniform() < 0.5:
                self.q1, self.q2 = self.q2, self.q1

    def perform_expert_trajectory(self):
        trajectory = self.env.expert_trajectory()
        total_reward = 0
        for i in range(len(trajectory) - 1):
            state, action, reward, next_state = *trajectory[i], trajectory[i + 1][0]
            total_reward += reward
            self.learn_from_step(state, action, reward, next_state)

        self._on_episode_end(total_reward)

    def perform_expert_trajectory_mc(self):
        trajectory = self.env.expert_trajectory()
        self.learn_from_episode(trajectory)

        total_reward = sum([r for _, _, r in trajectory if r is not None])
        self._on_episode_end(total_reward)

    def get_action(self, state):
        if not self.double_q:
            return np.argmax(self.q[state, :])
        else:
            return np.argmax(self.q1[state, :] + self.q2[state, :])

    @classmethod
    def load_from(cls, file):
        return pickle.load(file)

    @classmethod
    def save_to(cls, alg, file):
        pickle.dump(alg, file)


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
    def create_logger(cls):
        pass

    @classmethod
    @abstractmethod
    def stopping_cond(cls, test_reward):
        pass

    @classmethod
    def run(cls, env, args: argparse.Namespace):
        if not args.recodex and not args.evaluate and not args.save and not args.no_save:
            print("Specify --save or --no_save.")
            return

        if not args.evaluate_each and not args.evaluate:
            print("During training, --evaluate_each must be specified.")
            return

        if not args.model and args.evaluate:
            print("Supply --model for evaluation.")
            return

        # Set random seed
        np.random.seed(args.seed)

        if args.model:
            print(f"Loading the model from {args.model}.")
            algorithm = cls.get_algorithm_type().load_from_path(args.model)
        else:
            print("Initializing new model.")
            algorithm = cls.init_new_model(args)

        if args.evaluate:
            algorithm.init_evaluation(env)
            if not args.recodex:
                rewards = []
                while True:
                    reward = algorithm.evaluate_episode()
                    rewards.append(reward)
                    print(f"{len(rewards) - 1} : {reward}")
        else:
            cls._train(env, args, algorithm)
        if args.recodex:
            # Final evaluation
            while True:
                algorithm.evaluate_episode(final_evaluation=True)

    @classmethod
    def _train(cls, env, args, algorithm):
        if not args.recodex:
            logger = cls.create_logger()
            logger.begin(append=args.model is not None, force=args.force)
        else:
            logger = None

        config = DynamicConfig.from_dict(vars(args))
        if args.evaluate_each is not None:
            config.add("evaluate_now", AlternatingSchedule(args.evaluate_each))

        algorithm.init_training(env, config, logger, cls.stopping_cond)

        stopped_by_user = False

        def keyboard_input(inp):
            nonlocal stopped_by_user
            if inp == "stop":
                stopped_by_user = True

        keyboard_thread = KeyboardThread(keyboard_input)

        while not algorithm.should_stop_training:
            if stopped_by_user:
                print("Stopped by user.")
                break
            cls._do_training_step(args, algorithm)

        if not args.recodex:
            if args.save:
                print(f"Saving the model to {args.save}")
                cls.get_algorithm_type().save_to_path(algorithm, args.save)
            logger.end()

    @classmethod
    def _do_training_step(cls, args, algorithm):
        if args.pretrain:
            algorithm.perform_expert_trajectory()
        elif args.pretrain_mc:
            algorithm.perform_expert_trajectory_mc()
        elif args.mc:
            algorithm.perform_episode_mc()
        elif args.sarsa:
            algorithm.perform_episode_sarsa(n=4)
        else:
            algorithm.perform_episode()
