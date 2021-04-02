import numpy as np
import tensorflow as tf

from models.Model import Model
from training.TrainingCallback import TestCallback
from tensorflow.keras import layers, callbacks, optimizers, models


class Triplet(Model):
    def __init__(self, name, input_size, output_size, make_initial_preprocess):
        super().__init__(name, make_initial_preprocess)
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.create_model()

    @staticmethod
    def _batch_generator(X, y):
        data_length = X.shape[0]
        index = np.random.randint(data_length)
        anchor, anchor_y = X[index], y[index]

        positive = X[np.random.choice(np.where(y == anchor_y)[0])]
        negative = X[np.random.choice(np.where(y != anchor_y)[0])]

        return [list(anchor), list(positive), list(negative)]

    def _batches_generator(self, X, y, batch_size=32):
        """
        Array of batch_generator results

        batch_size - size of the generated array
        """
        all_data = np.array([self._batch_generator(X, y)
                             for _ in range(batch_size)])
        anchors = all_data[:, 0, :]
        positives = all_data[:, 1, :]
        negatives = all_data[:, 2, :]
        return anchors, positives, negatives

    def data_generator(self, X, y, batch_size=32):
        while True:
            batch = self._batches_generator(X, y, batch_size)
            labels = np.zeros((batch_size, self.input_size*3))
            yield batch, labels

    def triplet_loss(self, _, y_pred):
        alpha = 0.2
        y_pred = tf.convert_to_tensor(y_pred)

        anchor = y_pred[:, :self.output_size]
        positive = y_pred[:, self.output_size:2*self.output_size]
        negative = y_pred[:, 2*self.output_size:]

        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(positive_dist - negative_dist + alpha, 0.)

    def create_triplet_model(self):
        input_anchor = layers.Input(shape=self.input_size)
        input_positive = layers.Input(shape=self.input_size)
        input_negative = layers.Input(shape=self.input_size)

        model_anchor = self.model(input_anchor)
        model_positive = self.model(input_positive)
        model_negative = self.model(input_negative)

        result = layers.concatenate([model_anchor, model_positive,
                                     model_negative], axis=1)
        triplet = models.Model(
            [input_anchor, input_positive, input_negative], result)
        return triplet

    def train(self, batch_size: int = 128, epochs: int = 100,
              lr_patience: int = 3, stopping_patience: int = 12):

        X_train, x_test,  y_train, y_test = self.preprocess()

        triplet = self.create_triplet_model()

        steps_per_epoch = int(X_train.shape[0]/batch_size)
        optimizer = optimizers.Adam(0.1)
        lr_schedule = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                  patience=lr_patience,
                                                  min_delta=0.000001)

        early_stopping = callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,
                                                 patience=stopping_patience, restore_best_weights=True)

        # TODO: add TensorBoard
        # tb = callbacks.TensorBoard(log_dir="./tensor_board", histogram_freq=1)

        test_cb = TestCallback(X_train.reshape((-1, self.input_size)), y_train,
                               self.create_model(), input_size=self.input_size)

        triplet.compile(loss=self.triplet_loss, optimizer=optimizer)

        history = triplet.fit(self.data_generator(X_train, y_train, batch_size),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=1,
                              callbacks=[lr_schedule, early_stopping, test_cb])
        return history
