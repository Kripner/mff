#!/usr/bin/env python3
import argparse
import math
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import memory_game_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=4, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=16, type=int, help="Number of episodes to train on.")
parser.add_argument("--gradient_clipping", default=1.0, type=float, help="Gradient clipping.")
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
#parser.add_argument("--entropy_regularization", default=1, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=64, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=64, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--replay_buffer", default=64, type=int, help="Max replay buffer size; default batch_size")
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        # Define the agent inputs: a memory and a state.
        memory = tf.keras.layers.Input(shape=[args.memory_cells, args.memory_cell_size], dtype=tf.float32)
        state = tf.keras.layers.Input(shape=env.observation_space.shape, dtype=tf.int32)

        # Encode the input state, which is a (card, observation) pair,
        # by representing each element as one-hot and concatenating them, resulting
        # in a vector of length `sum(env.observation_space.nvec)`.
        encoded_input = tf.keras.layers.Concatenate()(
            [tf.one_hot(state[:, i], dim) for i, dim in enumerate(env.observation_space.nvec)])

        # TODO(memory_game): Generate a read key for memory read from the encoded input, by using
        # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
        # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
        # content in limited range).
        read_hidden = tf.keras.layers.Dense(args.hidden_layer, activation=tf.keras.activations.relu)(encoded_input)
        read_key = tf.keras.layers.Dense(args.memory_cell_size, activation=tf.keras.activations.tanh)(read_hidden)

        # TODO(memory_game): Read the memory using the generated read key. Notably, compute cosine
        # similarity of the key and every memory row, apply softmax to generate
        # a weight distribution over the rows, and finally take a weighted average of
        # the memory rows.
        similarities = tf.reduce_sum(tf.math.l2_normalize(memory, axis=1) * tf.math.l2_normalize(read_key), axis=2)
        read = tf.reduce_sum(memory * tf.math.softmax(similarities)[..., tf.newaxis], axis=1)

        # TODO(memory_game): Using concatenated encoded input and the read value, use a ReLU hidden
        # layer of size `args.hidden_layer` followed by a dense layer with
        # `env.action_space.n` units and `softmax` activation to produce a policy.
        policy_input = tf.keras.layers.Concatenate()([encoded_input, read])
        policy_hidden = tf.keras.layers.Dense(args.hidden_layer, activation=tf.keras.activations.relu)(policy_input)
        policy = tf.keras.layers.Dense(env.action_space.n, activation=tf.keras.activations.softmax)(policy_hidden)

        # TODO(memory_game): Perform memory write. For faster convergence, append directly
        # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
        # the last memory row to keep memory size constant.
        updated_memory = tf.concat([tf.expand_dims(encoded_input, axis=1), memory[:, :-1]], axis=1)

        # Create the agent
        self._agent = tf.keras.Model(inputs=[memory, state], outputs=[updated_memory, policy])
        self._agent.compile(
            optimizer=tf.optimizers.Adam(clipnorm=args.gradient_clipping),
            loss=tf.losses.SparseCategoricalCrossentropy(),
        )

    def zero_memory(self):
        # TODO(memory_game): Return an empty memory. It should be a TF tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]`.
        return tf.zeros([self.args.memory_cells, self.args.memory_cell_size])

    @tf.function
    def _train(self, states, actions, returns, episode_length):
        # TODO: Train the network given a batch of sequences of `states`
        # (each being a (card, symbol) pair), sampled `actions` and observed `returns`.
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `actions` as actions.
        #
        # Use the REINFORCE algorithm, optionally with a baseline. Note that
        # I use a baseline, but not a baseline computed by a neural network;
        # instead, for every time step, I track exponential moving average of
        # observed returns, with momentum 0.01. Furthermore, I use entropy regularization
        # with coefficient `args.entropy_regularization`.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        with tf.GradientTape() as tape:
            memory = tf.expand_dims(self.zero_memory(), axis=0)
            loss = 0
            for i in range(episode_length):
                s = tf.expand_dims(states[i], axis=0)
                memory, policy = self._agent([memory, s], training=True)
                loss += self._agent.loss(actions[i], policy) * returns[i]

                action_dist = tfp.distributions.Categorical(probs=policy[0])
                loss -= self.args.entropy_regularization * action_dist.entropy()

        grad = tape.gradient(loss, self._agent.trainable_variables)
        self._agent.optimizer.apply_gradients(zip(grad, self._agent.trainable_variables))

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        for episode in episodes:
            states, actions, returns = [], [], []
            for state, action, return_ in episode:
                states.append(state)
                actions.append(action)
                returns.append(return_)
            # Pad the arrays to a constant length with dummy
            for _ in range(2 * self.args.cards - len(episode)):
                states.append([0, 0])
                actions.append(0)
                returns.append(0)
            self._train(
                tf.convert_to_tensor(states),
                tf.convert_to_tensor(actions),
                tf.convert_to_tensor(returns),
                len(episode))

    @wrappers.typed_np_function(np.float32, np.int32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, memory, state):
        return self._agent([memory, state])


def main(env, args):
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    if args.replay_buffer is None:
        args.replay_buffer = args.batch_size
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], network.zero_memory()
        memory = tf.expand_dims(network.zero_memory(), axis=0)
        rewards, done = 0, False
        while not done:
            # TODO(memory_game): Find out which action to use
            memory, policy = network.predict(memory, tf.expand_dims(state, axis=0))
            action = np.argmax(policy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Training
    replay_buffer = wrappers.ReplayBuffer(max_length=args.replay_buffer)
    training = True
    action_space = np.arange(env.action_space.n)
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each):
            state, memory, episode, done = env.reset()[0], network.zero_memory(), [], False
            memory = tf.expand_dims(network.zero_memory(), axis=0)
            while not done:
                # TODO: Choose an action according to the generated distribution.
                memory, policy = network.predict(memory, tf.expand_dims(state, axis=0))
                if any([math.isnan(p) for p in policy[0]]):
                    print("NaN:", policy[0])
                action = np.random.choice(action_space, p=policy[0])

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode.append([state, action, reward])
                state = next_state

            # TODO: In the `episode`, compute returns from the rewards.
            G = 0
            for i in reversed(range(len(episode))):
                r = episode[i][2]
                G = G * args.gamma + r

                # Replace reward with return.
                episode[i][2] = G

            replay_buffer.append(episode)

            # Train the network if enough data is available
            if len(replay_buffer) >= args.batch_size:
                network.train(replay_buffer.sample(args.batch_size, np.random, replace=False))

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        if np.mean(returns) >= 1:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("MemoryGame-v0", cards=args.cards), args.seed, args.render_each,
                                 evaluate_for=args.evaluate_for, report_each=args.evaluate_for)

    main(env, args)
