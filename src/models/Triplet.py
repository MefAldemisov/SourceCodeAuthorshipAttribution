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
        # print("BATCH", batch.shape, labels.shape, indexes, files_with_labels, batch_size)
        return tf.convert_to_tensor(batch, np.float32), \
               tf.convert_to_tensor(labels.reshape(-1, 1), np.int32)

    def data_generator(self, X, y, batch_size=32):
        while True:
            yield self._batches_generator(X, y, batch_size)

    def dataset_generator(self, X, y, epochs, batch_size=32):
        dataset = tf.data.Dataset.from_generator(lambda: self.data_generator(X, y, batch_size=batch_size),
                                                 output_types=(tf.float32, tf.int32),
                                                 output_shapes=((batch_size, self.input_size), (batch_size, 1))
                                                 )
        dataset = dataset.repeat(epochs).batch(batch_size)
        print(dataset)
        return dataset

    @staticmethod
    def normalize(vector):
        """
        L2 norm of vector
        """
        return tf.math.divide_no_nan(vector,
                                     tf.sqrt(tf.reduce_sum(tf.square(vector))))

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
            start, end = self.normalize(start), self.normalize(end)
            product = tf.math.multiply(start, end)
            return product

        return None

    def hard_triplet_loss(self, y_true, y_pred):
        """
        :param y_pred: the distances, predicted for the triplet
        :return:
        """
        # select the array of dist(anchor, positive)
        # select the array of dist(anchor, negative)

        # TODO: made .self
        alpha = 0.5
        dist = "cos"
        batch_size = 1024
        print("Y true", y_true, y_true.shape)

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)

        # Compute pairwise distances
        ax_0 = tf.range(batch_size)
        ax_1 = tf.range(batch_size)

        grid = tf.meshgrid(ax_0, ax_1)
        stack = tf.stack(grid)
        indexes = tf.reshape(stack, (-1, 2))

        y_pred = tf.cast(y_pred, tf.dtypes.float32)
        distances = tf.map_fn(lambda a: self.get_distance(y_pred[a[0]], y_pred[a[1]], type=dist), indexes)

        y_true = tf.reshape(y_true, [batch_size, 1])
        y_y, y_x = tf.meshgrid(y_true, y_true)
        # print("grid", y_y.shape)
        # Select positive pairs (masking)
        equal_1 = tf.reshape(tf.math.equal(y_y, y_x), (-1, 1))
        equal = tf.reshape(tf.math.equal(y_y, y_x), (-1, 1)) # probably many errors and 2^50
        for i in range(1, self.output_size):
            equal = tf.concat([equal, equal_1], axis=1)
            # print("for", equal.shape)

        # print("EQ", equal.shape)
        # Select negative pairs
        n_equal = tf.math.logical_not(equal)

        print("Shape1", distances.shape, equal.shape)

        positive_dist = tf.boolean_mask(distances, equal)
        negative_dist = tf.boolean_mask(distances, n_equal)

        print("Shape2", positive_dist.shape, negative_dist.shape, type(alpha))
        print(positive_dist)
        return tf.maximum(positive_dist - negative_dist, 0)

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
                               self.create_model(), input_size=self.input_size, is_default=False)

        self.model.compile(loss=self.get_loss(), optimizer=optimizer)
        cbks = [lr_schedule, early_stopping, test_cb]

        h = self._batches_generator(X_train, y_train)
        history = self.model.fit(self.dataset_generator(np.array(X_train), np.array(y_train),
                                                        batch_size=batch_size, epochs=epochs),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                verbose=1,
                                callbacks=cbks)

        return history
