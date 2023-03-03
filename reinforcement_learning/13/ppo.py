#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import multi_collect_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=5, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.97, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=100, type=int, help="Steps for each worker to perform.")
parser.add_argument("--model", default="ppo.h5", type=str, help="Path to the model file.")


# TODO: Note that this time we derive the Network directly from `tf.keras.Model`.
# The reason is that the high-level Keras API is useful in PPO, where we need
# to train on an unchanging dataset (generated batches, train for several epochs, ...).
# That means that:
# - we define training in `train_step` method, which the Keras API automatically uses
# - we still provide custom `predict` method, because it is fastest this way
# - loading and saving should be performed using `save_weights` and `load_weights`, so that
#   the `predict` method and the `Network` type is preserved. If you use `.h5` suffix, the
#   checkpoint will be a single file, which is useful for ReCodEx submission.
class Network(tf.keras.Model):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        self.args = args

        # Create a suitable model for the given observation and action spaces.
        inputs = tf.keras.layers.Input(observation_space.shape)

        # TODO: Using a single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a policy with `action_space.n` discrete actions.
        policy_hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.keras.activations.relu)(inputs)
        policy = tf.keras.layers.Dense(action_space.n, activation=tf.keras.activations.softmax)(policy_hidden)

        # TODO: Using an independent single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a value function estimate. It is best to generate it as a scalar, not
        # a vector of length one, to avoid broadcasting errors later.
        value_hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.keras.activations.relu)(inputs)
        value = tf.keras.layers.Dense(1)(value_hidden)
        value = tf.squeeze(value)

        # Construct the model
        super().__init__(inputs=inputs, outputs=[policy, value])

        # Compile using Adam optimizer with the given learning rate.
        self.compile(optimizer=tf.optimizers.Adam(args.learning_rate))

    # TODO: Define a training method `train_step`, which is automatically used by Keras.
    def train_step(self, data):
        # Unwrap the data. The targets is a dictionary of several tensors, containing keys
        # - "actions"
        # - "action_probs"
        # - "advantages"
        # - "returns"
        states, targets = data
        actions, action_probs, advantages, returns = \
            targets["actions"], targets["action_probs"], targets["advantages"], targets["returns"]
        clip_epsilon = self.args.clip_epsilon
        with tf.GradientTape() as tape:
            # Compute the policy and the value function
            policy, value = self(states, training=True)

            # TODO: Sum the following three losses
            # - the PPO loss, where `self.args.clip_epsilon` is used to clip the probability ratio
            ratio = tf.divide(tf.gather_nd(policy, actions[..., tf.newaxis], batch_dims=1), action_probs)
            l_1 = tf.multiply(ratio, advantages)
            l_2 = tf.multiply(tf.clip_by_value(ratio, 1 - clip_epsilon, 1 + clip_epsilon), advantages)
            loss = -tf.reduce_mean(tf.minimum(l_1, l_2))
            # - the MSE error between the predicted value function and target regurns
            loss += tf.keras.losses.mean_squared_error(returns, value)
            # - the entropy regularization with coefficient `self.args.entropy_regularization`.
            #   You can compute it for example using `tf.losses.CategoricalCrossentropy()`
            #   by realizing that entropy can be computed using cross-entropy.
            entropy = tf.keras.losses.categorical_crossentropy(policy, policy)
            loss -= self.args.entropy_regularization * entropy

        # Perform an optimizer step and return the loss for reporting and visualization.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    # Predict method, with @wrappers.raw_tf_function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env.observation_space, env.action_space, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action using the greedy policy
            policy, _ = network.predict(np.array([state]))
            policy = policy[0]
            action = np.argmax(policy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    if args.recodex:
        network.load_weights(args.model)
        while True:
            evaluate_episode(start_evaluation=True)

    # Create the vectorized environment
    venv = gym.vector.make(env.spec.id, args.envs, asynchronous=True)

    # Training
    state = venv.reset(seed=args.seed)[0]
    training = True
    iteration = 0
    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.envs` actions, each
            # sampled from the corresponding policy generated by the `network.predict`
            # executed on the vector `state`.
            policy, value = network.predict(state)
            action = tf.squeeze(tf.random.categorical(tf.math.log(policy), 1))
            action_prob = policy[np.arange(args.envs), action]

            # Perform the step
            next_state, reward, terminated, truncated, _ = venv.step(action.numpy())
            done = terminated | truncated

            # TODO: Collect the required quantities
            states.append(state)
            actions.append(action)
            action_probs.append(action_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state

        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to process episodes of individual workers independently, and note that
        # each worker might have generated multiple episodes, the last one probably unfinished.
        advantages = np.zeros((args.worker_steps, args.envs), dtype=float)
        returns = np.zeros((args.worker_steps, args.envs), dtype=float)
        G = np.zeros(args.envs)
        for i in reversed(range(args.worker_steps)):
            # TODO: is this correct?! We should probably use one more gamma somewhere.
            G = G * args.trace_lambda * args.gamma * np.logical_not(dones[i]) + rewards[i]
            returns[i] = G
            advantages[i] = G - values[i]

        # Train using the Keras API.
        # - The below code assumes that the first two dimensions of the used quantities are
        #   `[args.worker_steps, args.envs]` and concatenates them together.
        # - We do not log the training by passing `verbose=0`; feel free to change it.
        network.fit(
            np.concatenate(states),
            {"actions": np.concatenate(actions),
             "action_probs": np.concatenate(action_probs),
             "advantages": np.concatenate(advantages),
             "returns": np.concatenate(returns)},
            batch_size=args.batch_size, epochs=args.epochs, verbose=0,
        )

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            eval_rewards = []
            for _ in range(args.evaluate_for):
                eval_rewards.append(evaluate_episode())
            if np.mean(eval_rewards) >= 475:
                network.save_weights(args.model)
                training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("SingleCollect-v0"), args.seed, args.render_each)

    main(env, args)
