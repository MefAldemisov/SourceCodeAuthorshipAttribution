from typing import List, Iterator

import numpy as np
import tensorflow as tf

from training.base.BaseTriplet import BaseTriplet


class SingleTriplet(BaseTriplet):

    def batch_generator(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        batch_size: int = 32):

        anchor_index = np.random.choice(y.shape[0], 1)
        positive_indexes, negative_indexes = self._positive_negative_index_generator(anchor_index, X, y,
                                                                                     batch_size=batch_size,
                                                                                     n_positive=batch_size // 2)

        X = X.reshape((-1, self.Model.input_size, 1))

        reduced_indexes = map(lambda x: np.random.choice(x, batch_size),
                              [positive_indexes, negative_indexes])

        positive, negative = map(lambda x: X[x], reduced_indexes)

        anchor = np.array([X[anchor_index] for _ in range(batch_size)]).reshape((batch_size,
                                                                                 self.Model.input_size,
                                                                                 1))

        return map(lambda x: tf.convert_to_tensor(x, dtype=tf.int32), [anchor, positive, negative])

    def triplet_loss(self,
                     args: List[tf.Tensor],
                     alpha: float = 0.2,
                     distance_metric: str = "euclidean"):
        anchor, positive, negative = args
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

    def loss_call(self,
                  data_generator: Iterator,
                  alpha: float,
                  distance_metric: str):
        triplets = list(next(data_generator))
        embeddings = list(map(lambda x: self.Model.model(x, training=True), triplets))
        loss = self.triplet_loss(embeddings, alpha=alpha, distance_metric=distance_metric)
        return loss
