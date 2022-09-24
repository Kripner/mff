#!/usr/bin/env python3
import argparse

import gym

import wrappers
from utils import *

LOG_DIR = "lander"
EXPERIMENT_ID = "train_3"

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
#parser.add_argument("--epsilon", default="lin{0:1,60000:0.1}", type=str, help="Exploration factor.")
parser.add_argument("--epsilon", default="lin{223000:0.1,250000:0.01}", type=str, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--init_optimistic", default=10, type=float, help="To what value to initialize value functions.")

add_universal_arguments(parser)


class LunarLanderTemplate(MainTemplate):
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
        return test_reward.value > 270


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    LunarLanderTemplate.run(env, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, args.render_each)

    main(env, args)
