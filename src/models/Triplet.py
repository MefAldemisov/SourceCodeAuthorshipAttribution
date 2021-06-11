from typing import Tuple

import tqdm
import numpy as np
import tensorflow as tf

from models.Model import Model
from training.TrainingCallback import TestCallback
from tensorflow.keras import optimizers, callbacks
# according to the documentation, BallTree is more efficient in high-dimensional case
from sklearn.neighbors import BallTree


class Triplet(Model):
    def __init__(self,
                 name: str,
                 input_size: int = 500,
                 output_size: int = 50,
                 make_initial_preprocess: bool = True):

        super().__init__(name, make_initial_preprocess)
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.create_model()
        self.index = None  # to create the index, the X.shape[0] value should be available

    def _positive_negative_index_generator(self,
                                           y_anchor: np.ndarray,
                                           X: np.ndarray,
                                           y: np.ndarray,
                                           n_positive: int,
                                           batch_size: int) -> Tuple:

        positive_indexes = np.where(y == y_anchor)[0][:n_positive]
        k = batch_size - positive_indexes.shape[0]

        if self.index is not None:
            anchor_index = np.random.choice(positive_indexes, 1)
            query = self.model.predict(X[anchor_index])
            query_res = self.index.query(query, batch_size, return_distance=False)[0]
            negative_indexes = np.array([neighbour_index for neighbour_index in query_res
                                         if y[neighbour_index] != y_anchor])[:k]
        else:  # the first batch generation
            negative_indexes = np.where(y != y_anchor)[0]
            np.random.shuffle(negative_indexes)
            negative_indexes = negative_indexes[:k]

        return positive_indexes, negative_indexes

    def _batches_generator(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           batch_size: int = 32):
        """
        The first author in the batch is selected randomly, then all of his files are selected to fit no more
        then half of the batch. All other elements are selected either randomly (if the index is empty), either
        the closes negative (for the anchor) examples are chosen.

        :param X, y - the overall dataset
        :param batch_size, int - size of the generated array

        :return two tensors:
        1. float32 tensor with source codes
        2. int32 tensor with the indexes of the authors
        """
        anchor_y = np.random.choice(y, 1)
        positive_indexes, negative_indexes = self._positive_negative_index_generator(anchor_y, X, y,
                                                                                     batch_size=batch_size,
                                                                                     n_positive=int(batch_size * 0.8))

        indexes = np.append(positive_indexes, negative_indexes)
        assert indexes.shape[0] == np.unique(indexes).shape[0]
        batch, labels = X[indexes], y[indexes]
        batch = tf.convert_to_tensor(batch, np.float32)
        labels = tf.convert_to_tensor(labels.reshape(-1, 1), np.int32)
        return batch, labels

    def _triplet_batch_generator(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           batch_size: int = 32):

        anchor_y = np.random.choice(y, 1)
        positive_indexes, negative_indexes = self._positive_negative_index_generator(anchor_y, X, y,
                                                                                     batch_size=batch_size,
                                                                                     n_positive=batch_size // 2)
        positive_indexes_indexes = np.random.choice(positive_indexes, batch_size)
        negative_indexes_indexes = np.random.choice(negative_indexes, batch_size)

        X = X.reshape((-1, self.input_size, 1))
        positive = X[positive_indexes_indexes]
        negative = X[negative_indexes_indexes]

        anchor = np.array([X[anchor_y] for _ in range(batch_size)]).reshape((batch_size, self.input_size, 1))
        return tf.convert_to_tensor(anchor), \
               tf.convert_to_tensor(positive), \
               tf.convert_to_tensor(negative)

    def data_generator(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       batch_size: int = 32):
        while True:
            yield self._triplet_batch_generator(X, y, batch_size)

    @staticmethod
    def get_distance(predictions: tf.Tensor,
                     metric: str = "euclidean"):
        """
        :param predictions: tensor, float32 - values of which pairwise distances will be computed
        :param metric: str in ["euclidean", "cos"|]
        :return: distance between `start` and `end` vectors
        """
        assert metric in ["euclidean", "cos"]
        if metric == "euclidean":
            # idea is taken from https://omoindrot.github.io/triplet-loss
            product = tf.matmul(predictions, tf.transpose(predictions))
            diagonal = tf.linalg.diag_part(product)
            distances = tf.expand_dims(diagonal, 0) + tf.expand_dims(diagonal, 1) - 2 * product
            return tf.maximum(distances, 0.0)

        elif metric == "cos":
            # angular distance(a, p) = 1 - a*p (element-wise)/sqrt(sum(square(a)))/sqrt(sum(square(p)))
            # definition of cos distance https://reference.wolfram.com/language/ref/CosineDistance.html?view=all
            predictions = tf.math.l2_normalize(predictions, axis=1)
            product = tf.maximum(1 - tf.matmul(predictions, tf.transpose(predictions)), .0)
            return product

        return None

    def hard_triplet_loss(self,
                          y_true: tf.Tensor,
                          y_pred: tf.Tensor,
                          alpha: float = 0.2,
                          distance_metric: str = "euclidean"):
        """
        :param y_true: labels of the source code authors
        :param y_pred: the distances, predicted for the triplet
        :param alpha: constant, margin of the triplet loss
        :param distance_metric: "euclidean" or "cos" - type of the distances to be used

        :return: triplet loss of the given predictions and labels
        triplet loss is defined according to the formula:
        `positive - negative + alpha`, where `positive` and `negative` are
        average distances between same/distinct-labeled predictions.

        In out case, triplet loss is defined as `e^(positive - negative + alpha)*positive`
        in order to make the function differentiable and non-zero valued (necessary for the visualization)
        """
        distances = self.get_distance(y_pred, metric=distance_metric)
        equal = tf.math.equal(tf.transpose(y_true), y_true)
        n_equal = tf.math.logical_not(equal)

        positive_dist = tf.reduce_mean(tf.boolean_mask(distances, equal))
        negative_dist = tf.reduce_mean(tf.boolean_mask(distances, n_equal))

        return tf.multiply(tf.maximum(tf.math.exp(positive_dist - negative_dist + alpha),
                                      10**(-9)), positive_dist) # don't to be zero

    def triplet_loss(self, anchor, positive, negative,
                     alpha: float = 0.2,
                     distance_metric: str = "euclidean"):
        # distance is euclidean
        norm = lambda x: tf.reduce_sum(tf.math.square(x), axis=1)
        if distance_metric == "euclidean":
            dist = lambda x, y: norm(x - y)
        elif distance_metric == "cos":
            dist = lambda x, y: tf.matmul(x / norm(x), y / norm(y))

        positive_distance = dist(anchor, positive)
        negative_distance = dist(anchor, negative)
        triplet = positive_distance - negative_distance + alpha
        return tf.reshape(triplet, (-1, 1))

    def on_batch_end(self, loss: tf.Tensor,
                     cbc: TestCallback,
                     epoch: int,
                     all_x: np.ndarray):
        """
        :param loss: loss of the batch training
        :param cbc: callback object
        :param epoch: int, index of the epoch
        :param all_x: x value to rebuild the tree
        """
        self.model.save("../outputs/{}.h".format(self.name))
        # get statistics
        loss_val = tf.keras.backend.get_value(loss)
        cbc.on_epoch_end(self.model, epoch, loss=loss_val)
        # update tree
        # predictions = self.model.predict(all_x)
        # self.index = BallTree(predictions, metric="euclidean")

    def training_loop(self, all_x: np.ndarray,
                      epochs: int,
                      steps_per_epoch: int,
                      data_generator,
                      optimizer: tf.keras.optimizers.Optimizer,
                      cbc: TestCallback,
                      alpha: float = 0.2,
                      distance_metric: str = "euclidean"):
        loss_function = self.triplet_loss

        for epoch in range(epochs):
            for _ in tqdm.tqdm(range(steps_per_epoch)):
                triplets = list(next(data_generator))
                with tf.GradientTape() as tape:
                    embeddings = map(lambda x: self.model(x, training=True), triplets)
                    loss = loss_function(*embeddings, alpha=alpha, distance_metric=distance_metric)
                    # update gradient
                    gradient = tape.gradient(loss, self.model.trainable_weights)
                    loss = tf.reduce_mean(loss, axis=0)[0]
                    optimizer.apply_gradients(zip(gradient, self.model.trainable_weights))

                assert np.isnan(self.model(triplets[-1]).numpy()).sum() == 0, "`nan` in the model's predictions"
                self.on_batch_end(loss, cbc, epoch, all_x)
            optimizer.lr = optimizer.lr * 0.9

    def train(self,
              batch_size: int = 64,
              epochs: int = 100,
              distance_metric: str = "euclidean",
              alpha: float = 0.1):

        X_train, x_test, y_train, y_test = self.preprocess()

        steps_per_epoch = int(X_train.shape[0] / batch_size)
        optimizer = optimizers.Adam(0.0001)

        test_cb = TestCallback(X_train, x_test, y_train, y_test,  threshold=alpha,
                               input_size=self.input_size, model_name=self.name, authors=[23, 34, 39, 40, 53, 60])

        tensorboard = callbacks.TensorBoard(log_dir="../outputs/tensor_board", histogram_freq=1)
        tensorboard.set_model(self.model)

        self.training_loop(X_train, epochs, steps_per_epoch,
                           self.data_generator(X_train, y_train, batch_size),
                           optimizer, test_cb, alpha=alpha,
                           distance_metric=distance_metric)
