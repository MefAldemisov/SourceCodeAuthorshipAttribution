from typing import List, Iterable

import numpy as np
import tensorflow as tf

from training.base.BaseTriplet import BaseTriplet


class AverageTriplet(BaseTriplet):

    def batch_generator(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        batch_size: int = 64):
        """
        The first author in the batch is selected randomly, then all of his files are selected to fit no more
        then half of the batch. All other elements are selected either randomly (if the index is empty), either
        the closes negative (for the anchor) examples are chosen.

        :param X - the data of the dataset
        :param y - the labels dataset
        :param batch_size, int - size of the generated array

        :return two tensors:
        1. float32 tensor with source codes
        2. int32 tensor with the indexes of the authors
        """
        anchor_index = np.random.choice(y.shape[0], 1)
        positive_indexes, negative_indexes = self._positive_negative_index_generator(anchor_index, X, y,
                                                                                     batch_size=batch_size,
                                                                                     n_positive=int(batch_size * 0.8))

        indexes = np.append(positive_indexes, negative_indexes)
        assert indexes.shape[0] == np.unique(indexes).shape[0]
        batch, labels = X[indexes], y[indexes]
        batch = tf.convert_to_tensor(batch, np.float32)
        labels = tf.convert_to_tensor(labels.reshape(-1, 1), np.int32)
        return batch, labels

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

    def triplet_loss(self,
                     args: List[tf.Tensor],
                     alpha: float = 0.2,
                     distance_metric: str = "euclidean") -> tf.Tensor:
        """
        :param args:
            element 1 - y_true: labels of the source code authors
            element 2 - y_pred: the distances, predicted for the triplet
        :param alpha: constant, margin of the triplet loss
        :param distance_metric: "euclidean" or "cos" - type of the distances to be used

        :return: triplet loss of the given predictions and labels
        triplet loss is defined according to the formula:
        `positive - negative + alpha`, where `positive` and `negative` are
        average distances between same/distinct-labeled predictions.

        In out case, triplet loss is defined as `e^(positive - negative + alpha)*positive`
        in order to make the function differentiable and non-zero valued (necessary for the visualization)
        """
        y_true, y_pred = args
        distances = self.get_distance(y_pred, metric=distance_metric)
        equal = tf.math.equal(tf.transpose(y_true), y_true)
        n_equal = tf.math.logical_not(equal)

        positive_dist = tf.reduce_mean(tf.boolean_mask(distances, equal))
        negative_dist = tf.reduce_mean(tf.boolean_mask(distances, n_equal))

        return tf.multiply(tf.maximum(tf.math.exp(positive_dist - negative_dist + alpha),
                                      10**(-9)), positive_dist)  # don't to be zero

    def loss_call(self, data_generator: Iterable, alpha: float, distance_metric: str):
        X, y = next(data_generator)
        embeddings = self.Model.model(X, training=True)
        loss = self.triplet_loss([y, embeddings], alpha=alpha, distance_metric=distance_metric)
        return loss
