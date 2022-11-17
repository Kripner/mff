#!/usr/bin/env python3
import argparse
from collections import deque

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a fixed seed
    generator = np.random.RandomState(args.seed)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=min(200, args.episodes))

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # The next action is always chosen in the epsilon-greedy way.
    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= args.epsilon else env.action_space.sample()
        return next_action, args.epsilon / env.action_space.n + (1 - args.epsilon) * (greedy_action == next_action)

    # The target policy is either the behavior policy (if not `args.off_policy`),
    # or the greedy policy (if `args.off_policy`).
    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    # Run the TD algorithm
    for _ in range(args.episodes):
        next_state, done = env.reset()[0], False
        history = []

        # Generate episode and update Q using the given TD method
        next_action, next_action_prob = choose_next_action(Q)
        while True:
            if not done:
                action, action_prob, state = next_action, next_action_prob, next_state

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if not done:
                    next_action, next_action_prob = choose_next_action(Q)
                history.append((state, action, action_prob, reward, next_state))
                assert len(history) <= args.n

            if len(history) == args.n or done:
                target_policy = compute_target_policy(Q)

                if args.mode == "sarsa":
                    G = 0
                    g_i = 1
                    for _, _, _, r, _ in history:
                        G += g_i * r
                        g_i *= args.gamma

                    rho = 1
                    if args.off_policy:
                        for s, a, a_prob, _, _, in history[1:]:
                            rho *= target_policy[s, a] / a_prob

                    if not done:
                        G += g_i * Q[next_state, next_action]
                        if args.off_policy:
                            rho *= target_policy[next_state, next_action] / next_action_prob

                    past_state, past_action, *_ = history.pop(0)
                    Q[past_state, past_action] += args.alpha * rho * (G - Q[past_state, past_action])

                    if done and len(history) == 0:
                        break
                elif args.mode == "expected_sarsa":
                    G = 0
                    g_i = 1
                    for _, _, _, r, _ in history:
                        G += g_i * r
                        g_i *= args.gamma

                    rho = 1
                    if args.off_policy:
                        for s, a, a_prob, _, _, in history[1:]:
                            rho *= target_policy[s, a] / a_prob

                    if not done:
                        G += g_i * np.dot(target_policy[next_state, :], Q[next_state, :])

                    past_state, past_action, *_ = history.pop(0)
                    Q[past_state, past_action] += args.alpha * rho * (G - Q[past_state, past_action])

                    if done and len(history) == 0:
                        break

                elif args.mode == "tree_backup":
                    G = 0
                    g_i = 1
                    for i in range(len(history)):
                        s, a, _, r, _ = history[i]

                        G += g_i * r
                        g_i *= args.gamma
                        if i != len(history) - 1:
                            n_s, n_a, _, _, _ = history[i + 1]
                            G += g_i * (np.dot(target_policy[n_s, :], Q[n_s, :]) - target_policy[n_s, n_a] * Q[n_s, n_a])
                            g_i *= target_policy[n_s, n_a]

                    if not done:
                        G += g_i * np.dot(target_policy[next_state, :], Q[next_state, :])

                    past_state, past_action, *_ = history.pop(0)
                    Q[past_state, past_action] += args.alpha * (G - Q[past_state, past_action])

                    if done and len(history) == 0:
                        break
                else:
                    raise Exception()


            # TODO: Perform the update to the state-action value function `Q`, using
            # a TD update with the following parameters:
            # - `args.n`: use `args.n`-step method
            # - `args.off_policy`:
            #    - if False, the epsilon-greedy behaviour policy is also the target policy
            #    - if True, the target policy is the greedy policy
            #      - for SARSA (with any `args.n`) and expected SARSA (with `args.n` > 1),
            #        importance sampling must be used
            # - `args.mode`: this argument can have the following values:
            #   - "sarsa": regular SARSA algorithm
            #   - "expected_sarsa": expected SARSA algorithm
            #   - "tree_backup": tree backup algorithm
            #
            # Perform the updates as soon as you can -- whenever you have all the information
            # to update `Q[state, action]`, do it. For each `action` use its corresponding
            # `action_prob` at the time of taking the `action` as the behaviour policy probability,
            # and the `compute_target_policy(Q)` with the current `Q` as the target policy.
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several state-action pairs that also need to be updated. Perform
            # the updates in the order in which you encountered the state-action
            # pairs and during these updates, use the `compute_target_policy(Q)`
            # with the up-to-date value of `Q`.

    return Q


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)