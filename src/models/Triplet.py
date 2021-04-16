import numpy as np
import tensorflow as tf

from models.Model import Model
import tensorflow_addons as tfa
from training.TrainingCallback import TestCallback
from tensorflow.keras import layers, callbacks, optimizers, models


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
            labels = None
            yield batch, labels

    def triplet_loss(self, _, y_pred):
        alpha = 0.2
        y_pred = tf.convert_to_tensor(y_pred)

        anchor = y_pred[:, :self.output_size]
        positive = y_pred[:, self.output_size:2 * self.output_size]
        negative = y_pred[:, 2 * self.output_size:]

        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(positive_dist - negative_dist + alpha, 0.)

    def get_distance(self, start, end, type="eucl"):
        """
        :param start:
        :param end:
        :param type: str in ["eucl", "cos"|]
        :return:
        """
        assert type in ["eucl", "cos"]
        if type == "eucl":
            return tf.square(start - end)
        elif type == "cos":
            # angular distance(a, p) = a*p (element-wise)/sqrt(sum(square(a)))/sqrt(sum(square(p)))
            # definition of cos distance https://reference.wolfram.com/language/ref/CosineDistance.html?view=all

            length_func = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x, axis=1)))
            div_func = lambda x, y: tf.math.divide_no_nan(x, y)

            product = tf.math.multiply(start, end, axis=1)
            l_start, l_end = length_func(start), length_func(end)
            distance = div_func(div_func(product, l_start), l_end)
            return distance

        return None

    def hard_triplet_loss(self, _, y_pred):
        """
        :param y_pred: the distances, predicted for the triplet
        :return:
        """
        # select the array of dist(anchor, positive)
        # select the array of dist(anchor, negative)
        alpha = 0.5
        selection_percent = 0.01
        dist = "cos"
        selection_num = selection_percent * self.output_size

        y_pred = tf.convert_to_tensor(y_pred)

        anchor = y_pred[:, :self.output_size]
        positive = y_pred[:, self.output_size:2 * self.output_size]
        negative = y_pred[:, 2 * self.output_size:]

        positive_dist = self.get_distance(anchor, positive, dist)
        negative_dist = self.get_distance(anchor, negative, dist)

        # sort both arrays
        # sorted_pos = tf.sort(positive_dist, axis=1, direction='DESCENDING')[:selection_num]
        # sorted_neg = tf.sort(negative_dist, axis=1, direction='ASCENDING')[:selection_num]

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

    def get_loss(self):
        if self.triplet_type == "default":
            return self.hard_triplet_loss
        elif self.triplet_type == "hard":
            return tfa.losses.TripletHardLoss(margin=0.2)
        elif self.triplet_type == "semi_hard":
            return tfa.losses.TripletSemiHardLoss(margin=0.2)
        else:
            raise ValueError("Invalid loss_type. Given {}, should be 'default', 'hard' or 'semi_hard'"
                             .format(self.triplet_type))

    # def add_input_layer(self):
    #     input_layer = layers.Input(shape=self.input_size)
    #     return models.Model(input_layer, self.model(input_layer))

    def train(self, batch_size: int = 128, epochs: int = 100,
              lr_patience: int = 3, stopping_patience: int = 12):

        X_train, x_test, y_train, y_test = self.preprocess()
        is_default = self.triplet_type == "default"
        triplet = self.create_triplet_model() if is_default else self.model
        steps_per_epoch = int(X_train.shape[0] / batch_size)
        optimizer = optimizers.Adam(0.1)
        lr_schedule = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                  patience=lr_patience,
                                                  min_delta=0.000001)

        early_stopping = callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,
                                                 patience=stopping_patience, restore_best_weights=True)

        # TODO: add TensorBoard
        # tb = callbacks.TensorBoard(log_dir="./tensor_board", histogram_freq=1)

        test_cb = TestCallback(X_train.reshape((-1, self.input_size)), y_train,
                               self.create_model(), input_size=self.input_size, is_default=is_default)

        triplet.compile(loss=self.get_loss(), optimizer=optimizer)
        cbks = [lr_schedule, early_stopping, test_cb]

        if is_default:
            history = triplet.fit(self.data_generator(X_train, y_train, batch_size),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=cbks)
        else:
            history = triplet.fit(X_train, y_train,
                                  validation_data=(x_test, y_test),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=cbks)
        return history
