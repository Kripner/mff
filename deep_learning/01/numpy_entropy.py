#!/usr/bin/env python3
import argparse
from collections import Counter
from typing import Tuple

import numpy as np
from scipy.special import xlogy

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_counts = Counter()
    data_total = 0
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_counts[line] += 1
            data_total += 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    # TODO: Load model distribution, each line `string \t probability`.
    model_values = {}
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            word, prob_str = line.split()
            prob = float(prob_str)
            assert word not in model_values
            model_values[word] = prob

    # TODO: Create a NumPy array containing the model distribution.
    all_words = list(set(model_values.keys()).union(data_counts.keys()))
    data = np.array([data_counts[w] / data_total for w in all_words])
    model = np.array([model_values.get(w, 0) for w in all_words])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = np.sum(-xlogy(data, data))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    crossentropy = np.sum(-data * np.log(model))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
