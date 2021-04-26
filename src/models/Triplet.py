import numpy as np
import tensorflow as tf
import tqdm

from models.Model import Model
from training.TrainingCallback import TestCallback
from tensorflow.keras import optimizers


class Triplet(Model):
    def __init__(self, name, triplet_type="default",
                 input_size=500, output_size=50,
                 make_initial_preprocess=True):
        super().__init__(name, make_initial_preprocess)
        self.triplet_type = triplet_type
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.create_model()

    @staticmethod
    def _batches_generator(X, y, batch_size=32):
        """
        Array of batch_generator results
        Selects a few persons in the dataset, then select the appropriate amount
        Of samples: while amount of files don't exceed the batch size
        - increase the amount of classes within a batch, the resulted number += batch_size//100

        batch_size - size of the generated array
        """
        possible_labels = []
        files_with_labels = 0

        while files_with_labels <= batch_size * 1.1:
            next_label = np.random.choice(y, 1)
            # don't to repeat the label:
            while next_label in possible_labels:
                next_label = np.random.choice(y)

            possible_labels.append(next_label)

            files_with_labels += len(y[np.where(y == next_label)])

        possible_labels = np.array(possible_labels)

        indexes = np.where(np.isin(y, possible_labels))[0]
        indexes = np.random.choice(indexes, batch_size)

        batch, labels = X[indexes], y[indexes]
        batch = tf.convert_to_tensor(batch, np.float32)
        labels = tf.convert_to_tensor(labels.reshape(-1, 1), np.int32)
        return batch, labels

    def data_generator(self, X, y, batch_size=32):
        while True:
            yield self._batches_generator(X, y, batch_size)

    @staticmethod
    def get_distance(predictions, metric="euclidean"):
        """
        :param predictions: tensor, float32 - values of which pairwise distances will be computed
        :param metric: str in ["euclidean", "cos"|]
        :return: distance between `start` and `end` vectors
        """
        assert metric in ["euclidean", "cos"]
        if metric == "euclidean":
            return tf.reduce_mean(tf.math.squared_difference(predictions,
                                                             tf.reshape(predictions, (-1, 1))))
        elif metric == "cos":
            # angular distance(a, p) = 1 - a*p (element-wise)/sqrt(sum(square(a)))/sqrt(sum(square(p)))
            # definition of cos distance https://reference.wolfram.com/language/ref/CosineDistance.html?view=all
            predictions = tf.math.l2_normalize(predictions, axis=1)  # self.normalize(end)
            product = tf.maximum(1 - tf.matmul(predictions, tf.transpose(predictions)), .0)
            return product

        return None

    def hard_triplet_loss(self, y_true, y_pred, alpha=0.2, distance_metric="euclidean"):
        """
        :param y_true: labels of the source code authors
        :param y_pred: the distances, predicted for the triplet
        :param alpha: constant, margin of the triplet loss
        :param distance_metric: "euclidean" or "cos" - type of the distances to be used

        :return: triplet loss of the given predictions and labels
        triplet loss is defined according to the formula:
        `positive - negative + alpha`, where `positive` and `negative` are
        average distances between same/distinct-labeled predictions.
        """
        distances = self.get_distance(y_pred, metric=distance_metric)

        equal = tf.math.equal(tf.transpose(y_true), y_true)
        n_equal = tf.math.logical_not(equal)

        positive_dist = tf.reduce_mean(tf.boolean_mask(distances, equal))
        negative_dist = tf.reduce_mean(tf.boolean_mask(distances, n_equal))

        return tf.maximum(positive_dist - negative_dist + alpha, .0)

    def training_loop(self, epochs, steps_per_epoch, data_generator, optimizer, cbc,
                      alpha=0.2, distance_metric="cos"):
        loss_function = self.hard_triplet_loss
        history = {"accuracy": [], "recall": [], "loss": []}
        for epoch in range(epochs):
            for step in tqdm.tqdm(range(steps_per_epoch)):
                x, y = next(data_generator)
                with tf.GradientTape() as tape:
                    predictions = self.model(x, training=True)
                    loss = loss_function(y, predictions, alpha=alpha,
                                         distance_metric=distance_metric)

                gradient = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(gradient, self.model.trainable_weights))

                self.model.save("../outputs/model.h")

                loss_val = tf.keras.backend.get_value(loss)
                accuracy, recall = cbc.on_epoch_end(self.model, step)
                print("Step:", step, "\t loss:", loss_val, "\t accuracy:", accuracy, "\t recall:", recall)

                history["loss"].append(loss_val)
                history["recall"].append(recall)
                history["accuracy"].append(accuracy)

        return history

    def train(self, batch_size: int = 128, epochs: int = 100,
              distance_metric="cos", alpha=0.1):

        X_train, x_test, y_train, y_test = self.preprocess()

        steps_per_epoch = int(X_train.shape[0] / batch_size)
        optimizer = optimizers.Adam(0.1)

        test_cb = TestCallback(X_train.reshape((-1, self.input_size)), y_train,
                               self.create_model(), input_size=self.input_size,
                               threshold=alpha)

        # self.model.run_eagerly = True
        history = self.training_loop(epochs, steps_per_epoch,
                                     self.data_generator(X_train, y_train, batch_size),
                                     optimizer, test_cb, alpha=alpha,
                                     distance_metric=distance_metric)
        return history
