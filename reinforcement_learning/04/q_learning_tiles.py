#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
parser.add_argument("--evaluate_each", default=200, type=int, help="Evaluate this often.")


# TODO: normalize the learning rate as by 1/t (t = number of tiles)

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon

    training = True
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        action = env.action_space.sample()
        while not done:
            q_state = np.sum(W[state], axis=0)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                W[state, action] += args.alpha * (reward - q_state[action])
                break

            q_next_state = np.sum(W[next_state], axis=0)
            next_action = env.action_space.sample() if np.random.uniform() < epsilon else np.argmax(q_next_state)
            W[state, action] += args.alpha * (reward + args.gamma * q_next_state[next_action] - q_state[action])

            state = next_state
            action = next_action

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        if env.episode % args.evaluate_each == 0:
            rewards = []
            for _ in range(15):
                state, done = env.reset()[0], False
                total_reward = 0
                while not done:
                    q_state = np.sum(W[state], axis=0)
                    action = np.argmax(q_state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                rewards.append(total_reward)
            avg_reward = np.average(rewards)
            print(f"Evaluation; average reward = {avg_reward}")
            if avg_reward > -105:
                training = False

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            q_state = np.sum(W[state], axis=0)
            action = np.argmax(q_state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=args.tiles),
                                 args.seed, args.render_each)

    main(env, args)
