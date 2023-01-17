#!/usr/bin/env python3
import argparse
import collections
import lzma
import os
import pickle
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers
import utils

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
#parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment.")
#parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--env", default="BipedalWalkerHardcore-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--parallel_envs_count", default=16, type=int, help="Environments.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--critic_learning_rate", default=0.0005, type=float, help="Critic learning rate.")
parser.add_argument("--actor_learning_rate", default=0.0003, type=float, help="Actor learning rate.")
parser.add_argument("--max_buffer_size", default=1_000_000, type=int, help="Maximum replay buffer size")
parser.add_argument("--min_buffer_size", default=5000, type=int, help="Minimum replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--train_every", default=8, type=int, help="Number of env steps between trainings.")

utils.add_universal_arguments(parser, default_directory="sac_walker")


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
                # self.sds_head = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.math.softplus)

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
                actions_distribution = tfp.bijectors.Scale((self.action_high - self.action_low) / 2)(
                    actions_distribution)
                actions_distribution = tfp.bijectors.Shift((self.action_high + self.action_low) / 2)(
                    actions_distribution)
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
        # self.target_entropy = args.target_entropy * env.action_space.shape[0]
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

    @staticmethod
    def assert_weights_equal(lhs, rhs):
        for net_a, net_b in zip(
                [lhs._actor, lhs._critic_A, lhs._target_critic_A, lhs._critic_B, lhs._target_critic_B],
                [rhs._actor, rhs._critic_A, rhs._target_critic_A, rhs._critic_B, rhs._target_critic_B]):
            for w1, w2 in zip(net_a.get_weights(), net_b.get_weights()):
                assert np.isclose(w1, w2).all()

    def save(self, path):
        assert path[-1] != '/'
        for network, name in self._get_networks():
            network.save_weights(path + "-" + name, save_format="h5")

    def load(self, path, simple_env):
        assert path[-1] != '/'

        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        self._actor(np.array([simple_env.observation_space.sample()]), sample=True)
        for critic in [self._critic_A, self._critic_B, self._target_critic_A, self._target_critic_B]:
            critic([np.array([simple_env.observation_space.sample()]), np.array([simple_env.action_space.sample()])])

        for network, name in self._get_networks():
            network.load_weights(path + "-" + name)

    def _get_networks(self):
        return [
            (self._actor, "actor"),
            (self._critic_A, "critic_A"),
            (self._target_critic_A, "target_critic_A"),
            (self._critic_B, "critic_B"),
            (self._target_critic_B, "target_critic_B"),
        ]


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])


class SACAlgorithm(utils.TemplateAlgorithm):
    def __init__(self, simple_env, args):
        super().__init__()

        self.generator = None
        self.action_space = None
        self.replay_buffer = None
        self.noise = None

        self.actions_num = simple_env.action_space.shape[0]
        self.network = Network(simple_env, args)

    def init_training(self, env, config, logger, evaluations_num):
        super().init_training(env, config, logger, evaluations_num)

        self.generator = np.random.RandomState(config.seed)
        self.action_space = np.arange(self.actions_num)
        self.replay_buffer = collections.deque(maxlen=args.max_buffer_size)

    def init_evaluation(self, env):
        super().init_evaluation(env)

        self.action_space = np.arange(self.actions_num)

    def __getstate__(self):
        return [self.network, self.actions_num, *super().__getstate__()]

    def __setstate__(self, state):
        self.network = state[0]
        self.actions_num = state[1]
        super().__setstate__(state[2:])

    def get_greedy_actions(self, states):
        return self.network.predict_mean_actions(states)

    def learn(self, cancellation_token: utils.CancellationToken):
        print("Training (SAC) ...")

        self.training = True
        states = self.env.reset()[0]
        while self.training:
            actions = self.network.predict_sampled_actions(states)

            next_states, rewards, terminated, truncated, _ = self.env.step(actions)
            done = np.logical_or(terminated, truncated)
            shaped_rewards = self._shape_rewards(rewards)

            assert self.envs_count == states.shape[0]
            for i in range(self.envs_count):
                self.replay_buffer.append(Transition(states[i], actions[i], shaped_rewards[i], done[i], next_states[i]))

            states = next_states
            self._on_frame_end(rewards, done)

            if len(self.replay_buffer) >= self.config.min_buffer_size and self.frames % self.config.train_every == 0:
                self._learn_from_buffer()

            if cancellation_token.is_cancelled():
                self.training = False

    def _shape_rewards(self, rewards):
        ended = rewards <= -100
        return ended * (rewards + 95) + np.logical_not(ended) * rewards

    def _learn_from_buffer(self):
        batch = np.random.randint(len(self.replay_buffer), size=self.config.batch_size)
        # batch = np.random.choice(len(self.replay_buffer), size=args.batch_size, replace=False)
        states, actions, rewards, dones, next_states = map(np.array, zip(*[self.replay_buffer[i] for i in batch]))

        next_values = self.network.predict_values(next_states)
        est_returns = rewards + self.config.gamma * np.logical_not(dones) * np.squeeze(next_values)
        self.network.train(states, actions, est_returns)

    def evaluate_episodes(self, num: Optional[int], final_evaluation=False,
                          on_ep_end_ckb=None) -> ([float], utils.ValueInInterval):
        assert num is None or num > 0

        if final_evaluation:
            states = self.env.reset(start_evaluation=True)[0]
        else:
            states = self.env.reset()[0]

        # TODO: work around this
        assert num is None or num >= self.envs_count
        started_num = self.envs_count
        results = []
        cumulative_rewards = np.zeros(self.envs_count)
        # To avoid bias, every episode that get started is also finished. Otherwise, a lot of very short episodes could
        # cause long ones to not be considered at all. Since environments are reset automatically, this mask says which
        # one are ignored, meaning they were started after the last considered episode.
        ignored = np.zeros(self.envs_count, dtype=bool)

        evaluating = True
        while evaluating:
            actions = self.get_greedy_actions(states)
            states, rewards, terminated, truncated, _ = self.env.step(actions)

            cumulative_rewards += rewards
            done = np.logical_or(terminated, truncated)

            for i in range(self.envs_count):
                if not ignored[i] and done[i]:
                    results.append(cumulative_rewards[i])
                    if on_ep_end_ckb is not None:
                        on_ep_end_ckb(cumulative_rewards[i])
                    cumulative_rewards[i] = 0

                    if num:
                        if started_num == num:
                            ignored[i] = True
                            if ignored.all():
                                evaluating = False
                        else:
                            started_num += 1

        result = utils.ValueInInterval(np.average(results), min(results), max(results))
        self.on_evaluation(result)
        return results, result

    def on_evaluation(self, result: utils.ValueInInterval):
        # TODO
        if result.value > 280:
            self.training = False

    @classmethod
    def save_to(cls, alg, filename: str):
        # TODO!!: also save self.done_episodes!
        alg.network.save(filename)

    @classmethod
    def load_from(cls, filename: str, simple_env=None) -> "SACAlgorithm":
        assert simple_env is not None
        alg = SACAlgorithm(simple_env, args)
        alg.network.load(filename, simple_env)
        return alg

    @classmethod
    def save_to_eval_only(cls, alg, filename):
        if not filename.endswith(".h5"):
            filename += ".h5"
        alg.network._actor.save_weights(filename, save_format="h5")

    @classmethod
    def load_from_eval_only(cls, filename, simple_env, args):
        alg = SACAlgorithm(simple_env, args)
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        alg.network._actor(np.array([simple_env.observation_space.sample()]), sample=True)
        alg.network._actor.load_weights(filename)
        return alg

    @staticmethod
    def test_save_and_load(simple_env, args, tmp_filename="tmp_test_save_000.model"):
        alg = WalkerTemplate.init_new_model(simple_env, args)
        # Init all the weights.
        alg.network._actor(np.array([simple_env.observation_space.sample()]), sample=False)
        for critic in [alg.network._critic_A, alg.network._critic_B, alg.network._target_critic_A,
                       alg.network._target_critic_B]:
            critic([np.array([simple_env.observation_space.sample()]), np.array([simple_env.action_space.sample()])])

        SACAlgorithm.save_to(alg, tmp_filename)
        loaded = SACAlgorithm.load_from(tmp_filename, simple_env)

        Network.assert_weights_equal(loaded.network, alg.network)
        assert loaded.done_episodes == alg.done_episodes

        # os.remove(tmp_filename)


class WalkerTemplate(utils.MainTemplate):
    @classmethod
    def get_algorithm_type(cls):
        return SACAlgorithm

    @classmethod
    def init_new_model(cls, simple_env, args: argparse.Namespace):
        return SACAlgorithm(simple_env, args)

    @classmethod
    def create_logger(cls, args):
        return utils.Logger(
            args,
            tracked_vars=["ep_idx", "train_reward", "test_reward"])

    @classmethod
    def get_vector_env(cls, args):
        vector_env = gym.vector.make(args.env, args.parallel_envs_count, asynchronous=True)
        if args.seed:
            vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
        return vector_env

    @classmethod
    def resolve_unknown_action(cls, action, simple_env, args) -> bool:
        if action == "test_saving":
            SACAlgorithm.test_save_and_load(simple_env, args)
            return True
        return False


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    if args.recodex:
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    print("LD_LIBRARY_PATH", os.environ.get('LD_LIBRARY_PATH', None))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # TODO: remove
#    args.model = "sac1.h5"

    WalkerTemplate.run(env, args)
    env.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
