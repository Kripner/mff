import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
num_classes = 10
image_shape = (28, 28)
validation_size = 5000

random_indices = np.random.permutation(train_x.shape[0])
train_indices, val_indices = random_indices[validation_size:], random_indices[:validation_size]
val_x, val_y = train_x[val_indices], train_y[val_indices]
train_x, train_y = train_x[train_indices], train_y[train_indices]

# Preprocessing
train_x = train_x.astype(float) / 255
val_x = val_x.astype(float) / 255
test_x = test_x.astype(float) / 255


def evaluate_hyperparameters(learning_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(image_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        train_x,
        train_y,
        batch_size=1,
        epochs=20,
        validation_data=(val_x, val_y),
    )

    val_loss, val_accuracy = model.evaluate(val_x, val_y)
    return model, val_accuracy, history.history


# Search for hyperparameters based on the validation set.
lr_options = [0.001, 0.01, 0.1]
best_model, best_lr, best_accuracy, best_history = None, None, float("-Inf"), None
for lr in lr_options:
    model, val_accuracy, history = evaluate_hyperparameters(learning_rate=lr)
    print(f"lr={lr}: validation accuracy = {val_accuracy}\n")
    if val_accuracy > best_accuracy:
        best_model, best_lr, best_accuracy, best_history = model, lr, val_accuracy, history
print(f"best learning rate based on validation data: {best_lr}")

# Plot the training.
fig, ax = plt.subplots()
ax.plot(best_history["loss"], color="C0")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")

ax2 = ax.twinx()
ax2.plot(best_history["val_sparse_categorical_accuracy"], color="C1")
ax2.set_ylabel("Validation accuracy")

plt.title(f"Training history (lr={best_lr})")
plt.savefig("history.png")
print("Training history saved to history.png.")

# Evaluate on test set.
test_loss, test_accuracy = best_model.evaluate(test_x, test_y)

print(f"accuracy on test data: {test_accuracy}")
