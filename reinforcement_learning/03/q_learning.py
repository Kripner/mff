#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers
from utils import *

LOG_DIR = "q_learning"
# EXPERIMENT_ID = "double_q;stopping;g=1;a=0.1;io=10"
EXPERIMENT_ID = "test"

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")

# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default="lin{0:0.5,7500:0.01}", type=str, help="Exploration factor.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--init_optimistic", default=10, type=float, help="To what value to initialize value functions.")

add_universal_arguments(parser)


class QLearningTemplate(MainTemplate):
    @classmethod
    def get_algorithm_type(cls):
        return QLearning

    @classmethod
    def init_new_model(cls, args):
        return QLearning(env, double_q=True, init_optimistic=args.init_optimistic)

    @classmethod
    def create_logger(cls):
        return Logger(
            LOG_DIR,
            EXPERIMENT_ID,
            tracked_vars=["ep_idx", "alpha", "epsilon", "gamma", "train_reward", "test_reward"])

    @classmethod
    def stopping_cond(cls, test_reward):
        return test_reward.value > -140


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    QLearningTemplate.run(env, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), args.seed, args.render_each)

    main(env, args)
