#!/usr/bin/env python3
from __future__ import annotations
import argparse
import collections
import math
import os
import queue
import threading
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, losses

from az_quiz import AZQuiz
import az_quiz_evaluator
import az_quiz_player_simple_heuristic
import az_quiz_player_fork_heuristic
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--load", default=False, action="store_true", help="Load a pretrained model before training.")
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz.model", type=str, help="Model path")
# TODO: run more simulations earlier in the game, then less
parser.add_argument("--num_simulations", default=100, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=1, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=1, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100000, type=int, help="Replay buffer max length.")
parser.add_argument("--min_window_length", default=100, type=int, help="Replay buffer min length.")


#########
# Agent #
#########
class Agent:
    def __init__(self, args: argparse.Namespace):
        # TODO: Define an agent network in `self._model`.
        #
        # A possible architecture known to work consists of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer with softmax
        #   activation to produce the policy,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.
        inp = layers.Input(shape=(7, 7, 4))
        h = inp
        for _ in range(5):
            h = layers.Conv2D(20, (3, 3), padding="same", activation="relu")(h)

        policy = layers.Conv2D(2, (3, 3), activation="relu")(h)
        policy = layers.Flatten()(policy)
        policy = layers.Dense(28, activation="softmax")(policy)
        policy = tf.squeeze(policy)

        value = layers.Conv2D(2, (3, 3), activation="relu")(h)
        value = layers.Flatten()(value)
        value = layers.Dense(1, activation="tanh")(value)
        value = tf.squeeze(value)

        self._model = models.Model(inputs=[inp], outputs=[policy, value])
        self._model.compile(
            loss=[losses.CategoricalCrossentropy(), losses.MeanSquaredError()],
            optimizer=optimizers.Adam(args.learning_rate)
        )
        self._model.summary()

    @classmethod
    def load(cls, path: str) -> Agent:
        # A static method returning a new Agent loaded from the given path.
        agent = Agent.__new__(Agent)
        agent._model = tf.keras.models.load_model(path)
        return agent

    def save(self, path: str, include_optimizer=True) -> None:
        # Save the agent model as a h5 file, possibly with/without the optimizer.
        self._model.save(path, include_optimizer=include_optimizer, save_format="h5")

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    #    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, boards: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> None:
        # TODO: Train the model based on given boards, target policies and target values.
        self._model.train_on_batch(boards, [target_policies, target_values])

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function.
        return self._model(boards)

    def board(self, game: AZQuiz) -> np.ndarray:
        # TODO: Generate the boards from the current AZQuiz game.
        #
        # The `game.board` returns a board representation, but you also need to
        # somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `AZQuiz.swap_players` might come handy);
        # - indicate the current player by adding channels to the representation.
        game = game.clone()
        if game.to_play == 1:
            game.swap_players()
        return game.board


########
# Utils #
########
def get_symmetries(board: np.ndarray):
    board = np.copy(board)
    rot = _rotate_board(board)
    rot_rot = _rotate_board(rot)
    return [board, rot, rot_rot, _flip_board(board), _flip_board(rot), _flip_board(rot_rot)]


def _flip_board(board: np.ndarray):
    flipped = np.zeros_like(board)
    for row in range(AZQuiz.N):
        for col in range(row + 1):
            flipped[row, row - col] = board[row, col]
    return flipped


def _rotate_board(board: np.ndarray):
    rotated = np.zeros_like(board)
    for row in range(AZQuiz.N):
        for col in range(row + 1):
            rotated[AZQuiz.N - row - 1 + col, AZQuiz.N - row - 1] = board[row, col]
    return rotated


def _get_random_board():
    game = AZQuiz(randomized=False)
    valid_actions = game.valid_actions()
    while len(valid_actions) > 0:
        game.move(np.random.choice(valid_actions))
        valid_actions = game.valid_actions()
    return game.board


def _print_boards(boards, labels, width=7):
    log = [[] for _ in range(8)]
    for i, (board, label) in enumerate(zip(boards, labels)):
        log[0].append(label.center(28))
        for row in range(7):
            log[1 + row].append("  " * (6 - row))
            for col in range(row + 1):
                log[1 + row].append(
                    " XX " if board[row, col, 0] else
                    " .. " if board[row, col, 1] else
                    " __ ")
            log[1 + row].append("  " * (6 - row))
        if len(log[0]) == width or i == len(boards) - 1:
            print(*["".join(line) for line in log], sep="\n")
            print()
            log = [[] for _ in range(8)]


########
# MCTS #
########
class MCTNode:
    def __init__(self, prior: Optional[float]):
        self.prior = prior  # Prior probability from the agent.
        self.game = None  # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0

    def value(self) -> float:
        # TODO: Return the value of the current node, handling the
        # case when `self.visit_count` is 0.

        if self.visit_count == 0:
            return float("-Inf")

        if self.game.winner is not None:
            return self.total_value

        return self.total_value / self.visit_count

    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # TODO: Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.
        if game.winner:
            value = 1 if game.winner == game.to_play else -1
        else:
            policy, value = agent.predict([agent.board(game)])
            for action, prob in zip(game.valid_actions(), policy):
                self.children[action] = MCTNode(prob)

        self.visit_count, self.total_value = 1, value

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # TODO: Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior

        noise = np.random.dirichlet([alpha] * len(self.children))
        for i, (action, child) in enumerate(self.children.items()):
            self.children[action].prior = epsilon * noise[i] + (1 - epsilon) * child.prior

    def select_child(self) -> tuple[int, MCTNode]:
        # Select a child according to the PUCT formula.
        def ucb_score(child):
            # TODO: For a given child, compute the UCB score as
            #   Q(s, a) + C(s) * P(s, a) * (sqrt(N(s)) / (N(s, a) + 1)),
            # where:
            # - Q(s, a) is the estimated value of the action stored in the
            #   `child` node. However, the value in the `child` node is estimated
            #   from the view of the player playing in the `child` node, which
            #   is usually the other player than the one playing in `self`,
            #   and in that case the estimated value must be "inverted";
            # - C(s) in AlphaZero is defined as
            #     log((1 + N(s) + 19652) / 19652) + 1.25
            #   Personally I used 1965.2 to account for shorter games, but I do not
            #   think it makes any difference;
            # - P(s, a) is the prior computed by the agent;
            # - N(s) is the number of visits of state `s`;
            # - N(s, a) is the number of visits of action `a` in state `s`.
            Q = -child.value()
            C = np.log((1 + self.visit_count + 1965) / 1965) + 1.25
            return Q + C * child.prior * (np.sqrt(self.visit_count) / (child.visit_count + 1))

        # TODO: Return the (action, child) pair with the highest `ucb_score`.
        return max(self.children.items(), key=lambda action_child: ucb_score(action_child[1]))


def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    # Run the MCTS search and return the policy proportional to the visit counts,
    # optionally including exploration noise to the root children.
    root = MCTNode(None)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Perform the `args.num_simulations` number of MCTS simulations.
    for _ in range(args.num_simulations):
        # TODO: Starting in the root node, traverse the tree using `select_child()`,
        # until a `node` without `children` is found.
        node = root
        parents = []
        last_action = None
        while len(node.children) != 0:
            parents.append(node)
            last_action, child = node.select_child()
            node = child

        # If the node has not been evaluated, evaluate it.
        if not node.is_evaluated():
            # TODO: Evaluate the `node` using the `evaluate` method. To that
            # end, create a suitable `AZQuiz` instance for this node by cloning
            # the `game` from its parent and performing a suitable action.
            updated_game = parents[-1].game.clone()
            updated_game.move(last_action)
            node.evaluate(updated_game, agent)
        else:
            # TODO: If the node has been evaluated but has no children, the
            # game ends in this node. Update it appropriately.
            node.visit_count += 1

        # Get the value of the node.
        value = node.value()

        # TODO: For all parents of the `node`, update their value estimate,
        # i.e., the `visit_count` and `total_value`.
        alternate = -1
        for parent in reversed(parents):
            parent.visit_count += 1
            parent.total_value += alternate * value
            alternate = -alternate

    # TODO: Compute a policy proportional to visit counts of the root children.
    # Note that invalid actions are not the children of the root, but the
    # policy should still return 0 for them.
    policy = np.zeros(game.actions)
    for action, child in root.children.items():
        policy[action] = child.visit_count / args.num_simulations
    return policy


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz(randomized=False)
    entries = []
    while game.winner is None:
        # TODO: Run the `mcts` with exploration.
        policy = mcts(game, agent, args, explore=True)

        # TODO: Select an action, either by sampling from the policy or greedily,
        # according to the `args.sampling_moves`.
        if len(entries) < args.sampling_moves:
            action = np.random.choice(game.actions, p=policy)
        else:
            action = np.argmax(policy)

        entries.append((agent.board(game), game.to_play, policy))

        game.move(action)

    # TODO: Return all encountered game states, each consisting of
    # - the board (probably via `agent.board`),
    # - the policy obtained by MCTS,
    # - the outcome based on the outcome of the whole game.
    states = []
    for board, to_play, policy in entries:
        outcome = 1 if game.winner == to_play else -1
        states.append(ReplayBufferEntry(board, policy, outcome))
    return states


def train(args: argparse.Namespace, cancel_token, agent=None) -> Agent:
    # Perform training
    if agent is None:
        agent = Agent(args)
    replay_buffer = wrappers.ReplayBuffer(max_length=args.window_length)
    generator = np.random.RandomState(args.seed)

    iteration = 0
    training = True
    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)

            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games:
                width = 7
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append("Move {}, result {}".format(i, outcome).center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                "{:>3.0f} ".format(policy[action] * 100))
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                    if len(log[0]) == width or i == len(game) - 1:
                        print(*["".join(line) for line in log], sep="\n")
                        print()
                        log = [[] for _ in range(8)]

        if len(replay_buffer) >= args.min_window_length:
            # Train
            for _ in range(args.train_for):
                # TODO: Perform training by sampling an `args.batch_size` of positions
                # from the `replay_buffer` and running `agent.train` on them.
                states = replay_buffer.sample(args.batch_size, generator)
                agent.train([e[0] for e in states], [e[1] for e in states], [e[2] for e in states])

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            score = az_quiz_evaluator.evaluate(
                [Player(agent, argparse.Namespace(num_simulations=0)), az_quiz_player_fork_heuristic.Player()],
                games=56, randomized=False, first_chosen=False, render=False, verbose=False)
            print("Evaluation after iteration {}: {:.1f}%".format(iteration, 100 * score), flush=True)
            if score > 0.95:
                training = False

        if cancel_token.is_cancelled():
            training = False

    return agent


#####################
# Evaluation Player #
#####################
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            policy, _ = self.agent.predict([self.agent.board(game)])
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = mcts(game, self.agent, self.args, explore=False)

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
class KeyboardThread(threading.Thread):
    def __init__(self, input_callback=None, name='keyboard_thread'):
        super(KeyboardThread, self).__init__(name=name)
        self.input_callback = input_callback
        self.setDaemon(True)
        self.start()

    def run(self):
        while True:
            self.input_callback(input())


class CancellationToken:
    def __init__(self):
        self.q = queue.Queue()

    def cancel(self):
        self.q.put(True)

    def is_cancelled(self):
        return not self.q.empty()


def main(args: argparse.Namespace) -> Player:
    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path)
    else:
        if args.load:
            agent = Agent.load(args.model_path)
            print("Loaded pretrained agent.")
        else:
            agent = None

        cancel_token = CancellationToken()

        def keyboard_input(inp):
            if inp == "stop":
                cancel_token.cancel()

        KeyboardThread(keyboard_input)

        # Perform training
        agent = train(args, cancel_token, agent)

        print("Saving the trained agent.")
        agent.save(args.model_path)

    return Player(agent, args)


def test_symmetries():
    b = _get_random_board()
    symmetries = get_symmetries(b)
    _print_boards(symmetries, ["original", "rot", "rot rot", "flip original", "flip rot", "flip rot rot"])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(args)

    if args.recodex:
        # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
        az_quiz_evaluator.evaluate(
            [player, az_quiz_player_simple_heuristic.Player()],
            games=56, randomized=False, first_chosen=False, render=False, verbose=True,
        )
