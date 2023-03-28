#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the implementation.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Convolution:
    def __init__(self, filters: int, kernel_size: int, stride: int, input_shape: List[int], output_shape: List[int], verify: bool) -> None:
        # Create a convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._verify = verify

        self._input_shape = input_shape
        self._output_shape = output_shape

        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(tf.initializers.GlorotUniform(seed=42)(
            [kernel_size, kernel_size, input_shape[2], filters]))
        self._bias = tf.Variable(tf.initializers.Zeros()([filters]))

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        # TODO: Compute the forward propagation through the convolution
        # with `tf.nn.relu` activation, and return the result.
        #
        # In order for the computation to be reasonably fast, you cannot
        # manually iterate through the individual pixels, batch examples,
        # input filters, or output filters. However, you can manually
        # iterate through the kernel size.
        m, n = inputs.shape[1], inputs.shape[2]
        out_m = int(tf.math.ceil((m - self._kernel_size + 1) / self._stride))
        out_n = int(tf.math.ceil((n - self._kernel_size + 1) / self._stride))
        output = tf.zeros((inputs.shape[0], out_m, out_n, self._filters))
        for i in range(self._kernel_size):
            for j in range(self._kernel_size):
                output += \
                    inputs[:, i:m+i-self._kernel_size+1:self._stride, j:n+j-self._kernel_size+1:self._stride]\
                    @ self._kernel[i, j]
        output += self._bias
        output = tf.nn.relu(output)

        # If requested, verify that `output` contains a correct value.
        if self._verify:
            reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            np.testing.assert_allclose(output, reference, atol=1e-4, err_msg="Forward pass differs!")

        return output

    def backward(
        self, inputs: tf.Tensor, outputs: tf.Tensor, outputs_gradient: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # TODO: Given this layer's inputs, this layer's outputs,
        # and the gradient with respect to the layer's outputs,
        # compute the derivatives of the loss with respect to
        # - the `inputs` layer,
        # - `self._kernel`,
        # - `self._bias`.
        inputs_gradient, kernel_gradient, bias_gradient = None, None, None

        G = tf.multiply(tf.cast(outputs > 0, tf.float32), outputs_gradient)

        bias_gradient = tf.reduce_sum(G, axis=[0, 1, 2])

        kernel_gradient = np.zeros(self._kernel.shape)
        for x in range(self._kernel_size):
            for y in range(self._kernel_size):
                x_bound = self._input_shape[0] - self._kernel_size + 1
                y_bound = self._input_shape[1] - self._kernel_size + 1
                inp = inputs[:, x:x + x_bound:self._stride, y:y + y_bound:self._stride, :]
                kernel_gradient[x, y] = tf.einsum("bxyc,bxyo->co", inp, G)
        kernel_gradient = tf.constant(kernel_gradient, dtype=tf.float32)

        inputs_gradient = tf.zeros(self._input_shape)
        start_padding = self._kernel_size - 1
        paddings = [
            [start_padding, self._input_shape[0] - (self._stride) * (self._output_shape[0] - 1) + self._output_shape[
                0] - start_padding],
            [start_padding, self._input_shape[1] - (self._stride) * (self._output_shape[1] - 1) + self._output_shape[
                1] - start_padding]]

        G_expand = self._expand_with_zeros(G, self._stride - 1, paddings)
        for x in range(self._kernel_size):
            for y in range(self._kernel_size):
                x_offset = self._kernel_size - 1 - x
                y_offset = self._kernel_size - 1 - y
                G_sel = G_expand[:, x_offset:x_offset + self._input_shape[0], y_offset:y_offset + self._input_shape[1],
                        :]
                inputs_gradient += tf.einsum("co,bxyo->bxyc", self._kernel[x, y, ...], G_sel)

        # If requested, verify that the three computed gradients are correct.
        if self._verify:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            for name, computed, reference in zip(
                    ["Inputs", "Kernel", "Bias"], [inputs_gradient, kernel_gradient, bias_gradient],
                    tape.gradient(reference, [inputs, self._kernel, self._bias], outputs_gradient)):
                np.testing.assert_allclose(computed, reference, atol=1e-4, err_msg=name + " gradient differs!")

        # Return the inputs gradient, the layer variables, and their gradients.
        return inputs_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]

    @staticmethod
    def _expand_with_zeros(batch, gaps, paddings):
        paddings = tf.constant([[0, 0]] + paddings + [[0, 0]])
        if gaps == 0:
            return tf.pad(batch, paddings)
        else:
            return tf.pad(Convolution._interleave_with_zeros(batch, gaps), paddings)

    @staticmethod
    def _interleave_with_zeros(batch, n):
        return tf.transpose(
            Convolution._interleave_with_zeros_horizontally(
                tf.transpose(
                    Convolution._interleave_with_zeros_horizontally(batch, n),
                    perm=(0, 2, 1, 3)),
                n),
            perm=(0, 2, 1, 3))

    @staticmethod
    def _interleave_with_zeros_horizontally(batch, n):
        batch_size = batch.shape[0]
        channel_count = batch.shape[3]
        new_shape = [batch_size, batch.shape[1], batch.shape[2] * (1 + n), channel_count]
        p = tf.reshape(
            tf.transpose(
                tf.pad(
                    tf.reshape(batch, [batch_size, 1, -1, channel_count]),
                    [[0, 0], [0, n], [0, 0], [0, 0]]),
                perm=(0, 2, 1, 3)
            ),
            new_shape)
        return p[:, :, :-n, :]


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convs = []
        for layer in args.cnn.split(","):
            filters, kernel_size, stride = map(int, layer.split("-"))
            output_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, filters]
            self._convs.append(Convolution(filters, kernel_size, stride, input_shape, output_shape, args.verify))
            input_shape = output_shape

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the metric and the optimizer
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate, jit_compile=False)

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            conv_values = [hidden]
            for conv in self._convs:
                hidden = conv.forward(hidden)
                conv_values.append(hidden)

            # Run the classification head
            hidden_flat = self._flatten(hidden)
            predictions = self._classifier(hidden_flat)

            # Compute the gradients of the classifier and the convolution output
            d_logits = (predictions - tf.one_hot(batch["labels"], MNIST.LABELS)) / len(batch["images"])
            variables = [self._classifier.bias, self._classifier.kernel]
            gradients = [tf.reduce_sum(d_logits, 0), tf.linalg.matmul(hidden_flat, d_logits, transpose_a=True)]
            hidden_gradient = tf.reshape(tf.linalg.matvec(self._classifier.kernel, d_logits), hidden.shape)

            # Backpropagate the gradient through the convolutions
            for conv, inputs, outputs in reversed(list(zip(self._convs, conv_values[:-1], conv_values[1:]))):
                hidden_gradient, conv_variables, conv_gradients = conv.backward(inputs, outputs, hidden_gradient)
                variables.extend(conv_variables)
                gradients.extend(conv_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self._accuracy.reset_states()
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"]
            for conv in self._convs:
                hidden = conv.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result()


def main(args: argparse.Namespace) -> float:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Load data, using only 5 000 training images
    mnist = MNIST(size={"train": 5_000})

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy))

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy))

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
