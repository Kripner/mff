#!/usr/bin/env python3
import argparse
import collections
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
#parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--envs", default=16, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=16, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--critic_learning_rate", default=0.001, type=float, help="Critic learning rate.")
parser.add_argument("--actor_learning_rate", default=0.0005, type=float, help="Actor learning rate.")
parser.add_argument("--model_path", default="walker.model", type=str, help="Model path")
parser.add_argument("--replay_buffer_size", default=1_000_000, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create an actor. Because we will be sampling (and `sample()` from
        # `tfp.distributions` does not play nice with functional models) and because
        # we need the `alpha` variable, we use subclassing to create the actor.
        class Actor(tf.keras.Model):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                # TODO: Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generating means with `env.action_space.shape[0]` units and no activation
                # - a layer for generating sds with `env.action_space.shape[0]` units and `tf.math.exp` activation
                # - finally, create a variable representing a logarithm of alpha, using for example the following:

                self.hidden_1 = tf.keras.layers.Dense(hidden_layer_size, activation="relu")
                self.hidden_2 = tf.keras.layers.Dense(hidden_layer_size, activation="relu")

                self.means_head = tf.keras.layers.Dense(env.action_space.shape[0])
                self.sds_head = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.math.exp)
                #self.sds_head = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.math.softplus)

                self._log_alpha = tf.Variable(np.log(0.1), dtype=tf.float32)

                self.action_low = env.action_space.low
                self.action_high = env.action_space.high

            def call(self, inputs: tf.Tensor, sample: bool):
                # TODO: Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                t = self.hidden_1(inputs)
                t = self.hidden_2(t)
                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                mus = self.means_head(t)
                sds = self.sds_head(t) if sample else tf.zeros_like(mus)
                # - Then, create the action distribution using `tfp.distributions.Normal`
                #   with the `mus` and `sds`. Note that to support computation without
                #   sampling, the easiest is to pass zeros as standard deviations when
                #   `sample == False`.
                actions_distribution = tfp.distributions.Normal(mus, sds)
                # - We then bijectively modify the distribution so that the actions are
                #   in the given range. Luckily, `tfp.bijectors` offers classes that
                #   can transform a distribution.
                #   - first run
                #       tfp.bijectors.Tanh()(actions_distribution)
                #     to squash the actions to [-1, 1] range,
                #   - then run
                #       tfp.bijectors.Scale((env.action_space.high - env.action_space.low) / 2)(actions_distribution)
                #     to scale the action ranges to [-(high-low)/2, (high-low)/2],
                #   - finally,
                #       tfp.bijectors.Shift((env.action_space.high + env.action_space.low) / 2)(actions_distribution)
                #     shifts the ranges to [low, high].
                #   In case you wanted to do this manually, sample from a normal distribution, pass the samples
                #   through the `tanh` and suitable scaling, and then compute the log-prob by using `log_prob`
                #   from the normal distribution and manually accounting for the `tanh` as shown in the slides.
                #   However, the formula from the slides is not numerically stable, for a better variant see
                #   https://github.com/tensorflow/probability/blob/ef1f64a434/tensorflow_probability/python/bijectors/tanh.py#L70-L81
                actions_distribution = tfp.bijectors.Tanh()(actions_distribution)
                actions_distribution = tfp.bijectors.Scale((self.action_high - self.action_low) / 2)(actions_distribution)
                actions_distribution = tfp.bijectors.Shift((self.action_high + self.action_low) / 2)(actions_distribution)
                # - Sample the actions by a `sample()` call.
                actions = actions_distribution.sample()
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                avg_log_probs = tf.reduce_mean(actions_distribution.log_prob(actions), axis=1)
                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                alpha = tf.math.exp(self._log_alpha)
                # - Return actions, log_prob, and alpha.
                return actions, avg_log_probs, alpha

        # TODO: Instantiate the actor as `self._actor` and compile it.
        self._actor = Actor(args.hidden_layer_size)
        self._actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.actor_learning_rate))

        # TODO: Create a critic, which
        # - takes observations and actions as inputs,
        # - concatenates them,
        # - passes the result through two dense layers with `args.hidden_layer_size` units
        #   and ReLU activation,
        # - finally, using a last dense layer produces a single output with no activation
        # This critic needs to be cloned so that two critics and two target critics are created.
        critic = self._create_critic(env, args)
        self._critic_A = tf.keras.models.clone_model(critic)
        self._target_critic_A = tf.keras.models.clone_model(critic)
        self._critic_B = tf.keras.models.clone_model(critic)
        self._target_critic_B = tf.keras.models.clone_model(critic)

        for c in [self._critic_A, self._critic_B]:
            c.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.critic_learning_rate),
                      loss=tf.keras.losses.MeanSquaredError())

        self.target_tau = args.target_tau
        # TODO: should the multiplication be there?
        #self.target_entropy = args.target_entropy * env.action_space.shape[0]
        self.target_entropy = args.target_entropy

    def _create_critic(self, env, args):
        obs_input = tf.keras.layers.Input(shape=env.observation_space.shape)
        actions_input = tf.keras.layers.Input(shape=env.action_space.shape)
        concat = tf.keras.layers.Concatenate()([obs_input, actions_input])
        hidden_1 = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(concat)
        hidden_2 = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(hidden_1)
        output = tf.keras.layers.Dense(1)(hidden_2)

        critic = tf.keras.models.Model(inputs=[obs_input, actions_input], outputs=output)
        return critic

    def save_actor(self, path: str):
        # Because we use subclassing for creating the actor, the easiest way of
        # serializing an actor is just to save weights.
        self._actor.save_weights(path, save_format="h5")

    def load_actor(self, path: str, env: wrappers.EvaluationEnv):
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        self.predict_mean_actions([env.observation_space.sample()])
        self._actor.load_weights(path)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Separately train:
        # - the actor, by using two objectives:
        #   - the objective for the actor itself; in this objective, `tf.stop_gradient(alpha)`
        #     should be used (for the `alpha` returned by the actor) to avoid optimizing `alpha`,
        #   - the objective for `alpha`, where `tf.stop_gradient(log_prob)` should be used
        #     to avoid computing gradient for other variables than `alpha`.
        #     Use `args.target_entropy` as the target entropy (the default of -1 per action
        #     component is fine and does not need to be tuned for the agent to train).
        self._train_actor(states)
        # - the critics using MSE loss.
        for critic, target_critic in zip([self._critic_A, self._critic_B],
                                         [self._target_critic_A, self._target_critic_B]):
            critic.optimizer.minimize(
                lambda: critic.compiled_loss(
                    returns,
                    critic([states, actions], training=True)),
                var_list=critic.trainable_variables
            )
            #
            # Finally, update the two target critic networks exponential moving
            # average with weight `args.target_tau`, using something like
            #   for var, target_var in zip(critic.trainable_variables, target_critic.trainable_variables):
            #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
            self._moving_average_update(critic, target_critic)

    def _train_actor(self, states):
        with tf.GradientTape() as actor_tape:
            actions, log_probs, alpha = self._actor(states, sample=True)
            values_a = self._critic_A([states, actions])
            values_b = self._critic_B([states, actions])
            values = tf.minimum(tf.squeeze(values_a), tf.squeeze(values_b))

            actor_loss = tf.reduce_mean(tf.stop_gradient(alpha) * log_probs - values)
            alpha_loss = -tf.reduce_mean(alpha * (tf.stop_gradient(log_probs) + self.target_entropy))
            loss = actor_loss + alpha_loss
        actor_grad = actor_tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(actor_grad, self._actor.trainable_variables))

    def _moving_average_update(self, network, target_network):
        for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)

    # Predict actions without sampling.
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_mean_actions(self, states: np.ndarray) -> np.ndarray:
        # Return predicted actions, assuming the actor is in `self._actor`.
        return self._actor(states, sample=False)[0]

    # Predict actions with sampling.
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_sampled_actions(self, states: np.ndarray) -> np.ndarray:
        # Return predicted actions, assuming the actor is in `self._actor`.
        return self._actor(states, sample=True)[0]

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Produce the predicted returns, which are the minimum of
        #    target_critic(s, a) - alpha * log_prob
        #  considering both target critics and actions sampled from the actor.
        actions, log_probs, alpha = self._actor(states, sample=True)
        values_a = self._target_critic_A([states, actions])
        values_b = self._target_critic_B([states, actions])
        return tf.minimum(tf.squeeze(values_a), tf.squeeze(values_b)) - alpha * log_probs


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action using the greedy policy.
            action = network.predict_mean_actions([state])[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        network.load_actor(args.model_path, env)
        while True:
            evaluate_episode(True)

    # Create the asynchronous vector environment for training.
    venv = gym.vector.make(args.env, args.envs, asynchronous=True)

    # Replay memory of a specified maximum size.
    replay_buffer = collections.deque(maxlen=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state, training = venv.reset(seed=args.seed)[0], True
    while training:
        for _ in range(args.evaluate_each):
            # Predict actions by calling `network.predict_sampled_actions`.
            action = network.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = venv.step(action)

            ended = reward <= -100
            shaped_rewards = ended * (reward + 95) + np.logical_not(ended) * reward
            done = terminated | truncated

            for i in range(args.envs):
                replay_buffer.append(Transition(state[i], action[i], shaped_rewards[i], done[i], next_state[i]))
            state = next_state

            # Training
            if len(replay_buffer) >= 4 * args.batch_size:
                # Note that until now we used `np.random.choice` with `replace=False` to generate
                # batch indices. However, this call is extremely slow for large buffers, because
                # it generates a whole permutation. With `np.random.randint`, indices may repeat,
                # but once the buffer is large, it happens with little probability.
                batch = np.random.randint(len(replay_buffer), size=args.batch_size)
                states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                # TODO: Perform the training
                next_values = network.predict_values(next_states)
                est_returns = rewards + args.gamma * np.logical_not(dones) * np.squeeze(next_values)
                network.train(states, actions, est_returns)

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        # TODO: when done, exit and save

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)