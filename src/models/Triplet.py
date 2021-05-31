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

    def _batches_generator(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           batch_size: int = 32):
        """
        Array of batch_generator results
        Selects a few persons in the dataset, then select the appropriate amount
        Of samples: while amount of files don't exceed the batch size
        - increase the amount of classes within a batch, the resulted number += batch_size//100

        batch_size - size of the generated array
        """
        anchor_y = np.random.choice(y, 1)
        positive_indexes = np.where(y == anchor_y)[0][:batch_size // 2]  # at least 50% of batch - other authors
        # negative indexes generation
        k = batch_size - positive_indexes.shape[0]

        if self.index is not None:
            anchor_index = np.random.choice(positive_indexes, 1)
            query = self.model.predict(X[anchor_index])
            query_res = self.index.query(query, batch_size, return_distance=False)[0]
            negative_indexes = np.array([neighbour_index for neighbour_index in query_res
                                         if y[neighbour_index] != anchor_y])[:k]
        else:  # the first batch generation
            negative_indexes = np.random.choice(y[np.where(y != anchor_y)[0]], k)

        indexes = np.append(positive_indexes, negative_indexes)
        batch, labels = X[indexes], y[indexes]
        batch = tf.convert_to_tensor(batch, np.float32)
        labels = tf.convert_to_tensor(labels.reshape(-1, 1), np.int32)
        return batch, labels

    def data_generator(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       batch_size: int = 32):
        while True:
            yield self._batches_generator(X, y, batch_size)

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
        """
        distances = self.get_distance(y_pred, metric=distance_metric)
        equal = tf.math.equal(tf.transpose(y_true), y_true)
        n_equal = tf.math.logical_not(equal)

        positive_dist = tf.reduce_mean(tf.boolean_mask(distances, equal))
        negative_dist = tf.reduce_mean(tf.boolean_mask(distances, n_equal))

        return positive_dist - negative_dist + alpha

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
        predictions = self.model.predict(all_x)
        self.index = BallTree(predictions, metric="euclidean")

    def training_loop(self, all_x: np.ndarray,
                      epochs: int,
                      steps_per_epoch: int,
                      data_generator,
                      optimizer: tf.keras.optimizers.Optimizer,
                      cbc: TestCallback,
                      lrs: tf.keras.callbacks.Callback,
                      alpha: float = 0.2,
                      distance_metric: str = "euclidean"):
        loss_function = self.hard_triplet_loss

        for epoch in range(epochs):
            for _ in tqdm.tqdm(range(steps_per_epoch)):
                x, y = next(data_generator)
                with tf.GradientTape() as tape:
                    predictions = self.model(x, training=True)
                    loss = loss_function(y, predictions, alpha=alpha,
                                         distance_metric=distance_metric)

                # update gradient
                gradient = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(gradient, self.model.trainable_weights))
                self.on_batch_end(loss, cbc, epoch, all_x)

            lrs.on_epoch_end(epoch)

    def train(self,
              batch_size: int = 64,
              epochs: int = 100,
              distance_metric: str = "euclidean",
              alpha: float = 0.1):

        X_train, x_test, y_train, y_test = self.preprocess()

        steps_per_epoch = int(X_train.shape[0] / batch_size)
        optimizer = optimizers.Adam(0.01)

        test_cb = TestCallback(X_train, x_test, y_train, y_test,  threshold=alpha,
                               input_size=self.input_size, model_name=self.name)
        lrs = callbacks.ReduceLROnPlateau(monitor="loss", factor=0.1, patience=2, min_lr=0.00001)

        tensorboard = callbacks.TensorBoard(log_dir="../outputs/tensor_board", histogram_freq=1)
        tensorboard.set_model(self.model)

        self.training_loop(X_train, epochs, steps_per_epoch,
                           self.data_generator(X_train, y_train, batch_size),
                           optimizer, test_cb, alpha=alpha,
                           distance_metric=distance_metric, lrs=lrs)
