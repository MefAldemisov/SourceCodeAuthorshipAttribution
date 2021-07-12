from training.TrainingCallback import TrainingCallback
from models.base.Model import Model
from models.data_processing.base.DataLoading import DataLoader
import numpy as np
import tqdm
import tensorflow as tf

from typing import Tuple, List, Iterable
from tensorflow.keras import optimizers, callbacks
# according to the documentation, BallTree is more efficient in high-dimensional case
from sklearn.neighbors import BallTree


class BaseTriplet:
    def __init__(self, model: Model and DataLoader):
        self.index = None
        self.Model = model

    def _positive_negative_index_generator(self,
                                           anchor_index: int,
                                           X: np.ndarray,
                                           y: np.ndarray,
                                           n_positive: int,
                                           batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        y_anchor = y[anchor_index]
        positive_indexes = np.where(y == y_anchor)[0][:n_positive]
        k = batch_size - positive_indexes.shape[0]

        if self.index is not None:
            query = self.Model.model.predict(X[anchor_index])
            query_res = self.index.query(query, 5*batch_size, return_distance=False)[0]
            negative_indexes = np.array([neighbour_index for neighbour_index in query_res
                                         if y[neighbour_index] != y_anchor])[:k]
        else:  # the first batch generation
            negative_indexes = np.where(y != y_anchor)[0]
            np.random.shuffle(negative_indexes)
            negative_indexes = negative_indexes[:k]

        return positive_indexes, negative_indexes

    def batch_generator(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        batch_size: int):
        raise NotImplementedError

    def data_generator(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       batch_size: int):
        while True:
            yield self.batch_generator(X, y, batch_size)

    def triplet_loss(self, args: List, alpha: float, distance_metric: str) -> tf.Tensor:
        raise NotImplementedError

    def on_batch_end(self, loss: tf.Tensor,
                     cbc: TrainingCallback,
                     epoch: int,
                     all_x: np.ndarray,
                     step: int):
        """
        :param loss: loss of the batch training
        :param cbc: callback object
        :param epoch: int, index of the epoch
        :param all_x: x value to rebuild the tree
        """
        self.Model.model.save("../outputs/{}_{}.h".format(self.Model.name, epoch))
        # get statistics
        if step % 10 == 0:
            loss_val = tf.keras.backend.get_value(loss)
            cbc.on_epoch_end(self.Model.model, epoch, loss=loss_val)
        # update tree
        predictions = self.Model.model.predict(all_x)
        self.index = BallTree(predictions, metric="euclidean")

    def loss_call(self, data_generator: Iterable, alpha: float, distance_metric: str):
        raise NotImplementedError

    def training_loop(self, all_x: np.ndarray,
                      epochs: int,
                      steps_per_epoch: int,
                      data_generator: Iterable,
                      optimizer: tf.keras.optimizers.Optimizer,
                      cbc: TrainingCallback,
                      alpha: float = 0.2,
                      distance_metric: str = "euclidean"):

        for epoch in range(epochs):
            for step in tqdm.tqdm(range(steps_per_epoch)):
                with tf.GradientTape() as tape:
                    loss = self.loss_call(data_generator, alpha, distance_metric)
                    # update gradient
                    gradient = tape.gradient(loss, self.Model.model.trainable_weights)
                    optimizer.apply_gradients(zip(gradient, self.Model.model.trainable_weights))
                # triplets = list(next(data_generator))
                # assert np.isnan(self.Model.model(triplets[-1]).numpy()).sum() == 0, "`nan` in the model's predictions"

                loss = tf.reduce_mean(loss, axis=0)[0]
                self.on_batch_end(loss, cbc, epoch, all_x, step)
            optimizer.lr = optimizer.lr * 0.9

    def train(self,
              batch_size: int = 64,
              epochs: int = 100,
              distance_metric: str = "euclidean",
              alpha: float = 0.1):

        X_train, x_test, y_train, y_test = self.Model.preprocess()

        steps_per_epoch = int(X_train.shape[0] / batch_size)
        optimizer = optimizers.Adam(0.0001)

        test_cb = TrainingCallback(X_train, x_test, y_train, y_test, threshold=alpha,
                                   input_size=self.Model.input_size,
                                   model_name=self.Model.name, authors=[23, 34, 39, 40, 53, 60])

        tensorboard = callbacks.TensorBoard(log_dir="../outputs/tensor_board", histogram_freq=1)
        tensorboard.set_model(self.Model.model)

        self.training_loop(X_train, epochs, steps_per_epoch,
                           self.data_generator(X_train, y_train, batch_size),
                           optimizer, test_cb, alpha=alpha,
                           distance_metric=distance_metric)
