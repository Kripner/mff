#!/usr/bin/env python3
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=4000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor.")
parser.add_argument("--measure_each", default=0, type=int, help="Measure performance after some episodes.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Set random seed
    np.random.seed(args.seed)

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros([num_states, num_actions], dtype=float)
    C = np.zeros([num_states, num_actions], dtype=int)

    measured_returns = []
    for e in range(args.episodes):
        measuring = args.measure_each and e % args.measure_each == 0
        records = []
        state, done = env.reset()[0], False
        while not done:
            if measuring or np.random.uniform() >= args.epsilon:
                action = np.argmax(Q[state, :])
            else:
                action = env.action_space.sample()

            # Perform the action.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            records.append((state, action, reward))
            state = next_state

        if measuring:
            measured_returns.append(env.episode_returns[-1])

        q = 0
        for s, a, r in reversed(records):
            #TODO: this is weird - actions done near the 500 mark are considered bad
            # We might instead choose appropriate return estimate for the final action and use the discount factor
            q = q * args.gamma + r
            C[s, a] += 1
            Q[s, a] += 1 / C[s, a] * (q - Q[s, a])

    if args.measure_each:
        plt.plot(np.arange(0, env.episode - 1, args.measure_each), measured_returns)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        print("Saving returns.png")
        plt.savefig("returns.png")

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed, args.render_each)

    main(env, args)
