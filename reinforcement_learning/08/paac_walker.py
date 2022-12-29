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
import utils

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
# parser.add_argument("--entropy_regularization_decay", default=1, type=float, help="Entropy regularization decay.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=20, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--workers", default=16, type=int, help="Number of parallel workers.")

utils.add_universal_arguments(parser, default_directory="sac_walker")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        actor_mus = tf.keras.models.Sequential([
            # tf.keras.layers.Embedding(env.observation_space.nvec[-1], 128, input_length=args.tiles),
            tf.keras.layers.CategoryEncoding(input_shape=(args.tiles,), num_tokens=env.observation_space.nvec[-1],
                                             output_mode="multi_hot"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            # TODO: scale tf.tanh to make it more general (even if it wasn't the case that low == -1, high == 1.
            # -> low + (high - low) * ((tanh(x) + 1) / 2)
            tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.tanh)
        ])
        actor_mus.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanSquaredError()])
        self._actor_mus = actor_mus
        actor_mus.summary()

        actor_sds = tf.keras.models.Sequential([
            tf.keras.layers.CategoryEncoding(input_shape=(args.tiles,), num_tokens=env.observation_space.nvec[-1],
                                             output_mode="multi_hot"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.nn.softplus)
        ])
        actor_sds.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanSquaredError()])
        self._actor_sds = actor_sds
        actor_sds.summary()

        critic = tf.keras.models.Sequential([
            tf.keras.layers.CategoryEncoding(input_shape=(args.tiles,), num_tokens=env.observation_space.nvec[-1],
                                             output_mode="multi_hot"),
            tf.keras.layers.Dense(args.hidden_layer_size, activation="relu", input_shape=env.observation_space.shape),
            tf.keras.layers.Dense(1)
        ])
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.MeanSquaredError()])
        self._critic = critic
        critic.summary()

        self.entropy_reg = args.entropy_regularization

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Run the model on given `states` and compute
        # `sds`, `mus` and predicted values. Then create `action_distribution` using
        # `tfp.distributions.Normal` class and the computed `mus` and `sds`.
        # In PyTorch, the corresponding class is `torch.distributions.Normal`.
        #
        # TODO: Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - predicted values)`
        #   and compute its mean. Note that the gradient must not flow through
        #   the predicted values (you can use `tf.stop_gradient` if necessary).
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.

        values = self._critic(states)
        with tf.GradientTape(persistent=True) as tp:
            mus = self._actor_mus(states, training=True)
            sds = self._actor_sds(states, training=True)
            action_dist = tfp.distributions.Normal(mus, sds)

            loss = - tf.reduce_mean(tf.reduce_sum(action_dist.log_prob(actions), axis=1) * (returns - values)) \
                   - self.entropy_reg * action_dist.entropy()

        mus_grad = tp.gradient(loss, self._actor_mus.trainable_variables)
        self._actor_mus.optimizer.apply_gradients(zip(mus_grad, self._actor_mus.trainable_variables))

        sds_grad = tp.gradient(loss, self._actor_sds.trainable_variables)
        self._actor_sds.optimizer.apply_gradients(zip(sds_grad, self._actor_sds.trainable_variables))
        # self._actor_mus.optimizer.minimize(
        #     loss,
        #     var_list=self._actor_mus.trainable_variables
        # )
        # self._actor_sds.optimizer.minimize(
        #     loss,
        #     var_list=self._actor_sds.trainable_variables
        # )

        self._critic.optimizer.minimize(
            lambda: self._critic.compiled_loss(returns, self._critic(states)),
            var_list=self._critic.trainable_variables
        )

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return predicted action distributions (mus and sds).
        return self._actor_mus(states), self._actor_sds(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted state-action values.
        return self._critic(states)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])


class A3CAlgorithm(utils.TemplateAlgorithm):
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
        # self.noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
        self.noise = OrnsteinUhlenbeckNoise(env.action_space.shape, 0, args.noise_theta, args.noise_sigma)

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
        mus, _ = network.predict_actions(states)
        return mus

    def learn(self, cancellation_token: utils.CancellationToken):
        print("Training (TD3) ...")

        self.training = True
        states = self.env.reset()[0]
        while self.training:
            mus, sds = network.predict_actions(states)
            actions = np.random.normal(mus, sds)
            # TODO: use noise?
            # actions += self.noise.sample()
            actions = np.clip(actions, env.action_space.low, env.action_space.high)

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = np.logical_or(terminated, truncated)
            shaped_rewards = self._shape_rewards(rewards)

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            est_next_values = np.squeeze(network.predict_values(next_states))
            est_returns = shaped_rewards + np.logical_not(terminated) * est_next_values

            # TODO(paac): Train network using current states, chosen actions and estimated returns
            network.train(states, actions, est_returns)

            states = next_states
            self._on_frame_end(rewards, done)

            if cancellation_token.is_cancelled():
                self.training = False

    def _shape_rewards(self, rewards):
        ended = rewards <= -100
        return ended * (rewards + 95) + np.logical_not(ended) * rewards

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

                    if num:
                        if started_num == num:
                            ignored[i] = True
                            if ignored.all():
                                evaluating = False
                        else:
                            cumulative_rewards[i] = 0
                            started_num += 1

        result = utils.ValueInInterval(np.average(results), min(results), max(results))
        self.on_evaluation(result)
        return results, result

    def on_evaluation(self, result: utils.ValueInInterval):
        # TODO
        # if result.lower == 500:
        #    self.training = False
        pass

    @classmethod
    def save_to(cls, alg, filename: str):
        with lzma.open(filename, "wb") as model_file:
            pickle.dump(alg, model_file)

    @classmethod
    def load_from(cls, filename: str) -> "A3CAlgorithm":
        with lzma.open(filename, "rb") as model_file:
            return pickle.load(model_file)

    #    @classmethod
    #    def save_to_eval_only(cls, alg, filename):
    #        alg.network.save(filename + ".h5", include_optimizer=False, save_traces=False)
    #
    #    @classmethod
    #    def load_from_eval_only(cls, filename):
    #        alg = DQNAlgorithm.__new__(DQNAlgorithm)
    #        net = models.load_model(filename + ".h5")
    #        alg.network = Network(net)
    #        return alg

    @classmethod
    def save_to_eval_only(cls, alg, filename):
        if not filename.endswith(".h5"):
            filename += ".h5"
        alg.network.actor.save_weights(filename, save_format="h5")

    @classmethod
    def load_from_eval_only(cls, filename, simple_env, args):
        alg = A3CAlgorithm(simple_env, args)
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        alg.network.predict_actions(np.array([simple_env.observation_space.sample()]))
        alg.network.actor.load_weights(filename)
        return alg

    @staticmethod
    def test_save_and_load(alg: "A3CAlgorithm", tmp_filename="tmp_test_save_000.model"):
        A3CAlgorithm.save_to(alg, tmp_filename)
        loaded = A3CAlgorithm.load_from(tmp_filename)

        Network.assert_weights_equal(loaded.network, alg.network)
        assert loaded.done_episodes == alg.done_episodes

        os.remove(tmp_filename)


class WalkerTemplate(utils.MainTemplate):
    @classmethod
    def get_algorithm_type(cls):
        return A3CAlgorithm

    @classmethod
    def init_new_model(cls, simple_env, args: argparse.Namespace):
        return A3CAlgorithm(simple_env, args)

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
            alg = cls.init_new_model(simple_env, args)
            A3CAlgorithm.test_save_and_load(alg)
            return True
        return False


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    if args.recodex:
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    WalkerTemplate.run(env, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles),
        args.seed, args.render_each)

    main(env, args)
