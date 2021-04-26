import numpy as np
import tensorflow as tf
import tqdm

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
        batch = tf.convert_to_tensor(batch, np.float32)
        labels = tf.convert_to_tensor(labels.reshape(-1, 1), np.int32)
        return batch, labels

    def data_generator(self, X, y, batch_size=32):
        while True:
            yield self._batches_generator(X, y, batch_size)

    def dataset_generator(self, X, y, epochs, batch_size=32):
        dataset = tf.data.Dataset.from_generator(lambda: self.data_generator(X, y, batch_size=batch_size),
                                                 output_types=(tf.float32, tf.int32),
                                                 output_shapes=((batch_size, self.input_size), (batch_size, 1))
                                                 )
        dataset = dataset.repeat(epochs).batch(1)
        print(dataset)
        return dataset

    @staticmethod
    def normalize(vector):
        """
        L2 norm of vector
        """
        return tf.math.divide_no_nan(vector,
                                     tf.sqrt(tf.reduce_sum(tf.square(vector))))

    def get_distance(self, start, end, metric="euclidean"):
        """
        :param start: tensor, float32
        :param end: tensor, float32
        :param metric: str in ["euclidean", "cos"|]
        :return: distance between `start` and `end` vectors
        """
        assert metric in ["euclidean", "cos"]
        if metric == "euclidean":
            return tf.square(start - end)
        elif metric == "cos":
            # angular distance(a, p) = a*p (element-wise)/sqrt(sum(square(a)))/sqrt(sum(square(p)))
            # definition of cos distance https://reference.wolfram.com/language/ref/CosineDistance.html?view=all
            start, end = self.normalize(start), self.normalize(end)
            product = tf.reduce_sum(tf.math.multiply(start, end))
            return product

        return None

    def hard_triplet_loss(self, y_true, y_pred):
        """
        :param y_true: labels of the source code authors
        :param y_pred: the distances, predicted for the triplet
        :return:
        """
        # select the array of dist(anchor, positive)
        # select the array of dist(anchor, negative)

        # TODO: made .self
        alpha = 0.1
        dist = "cos"
        batch_size = 128

        # Compute pairwise distances
        ax_0 = tf.range(batch_size)
        ax_1 = tf.range(batch_size)

        grid = tf.meshgrid(ax_0, ax_1)
        stack = tf.stack(grid, axis=2)
        indexes = tf.reshape(stack, (-1, 2))

        distances = tf.map_fn(lambda a: self.get_distance(y_pred[a[0]], y_pred[a[1]], metric=dist),
                              indexes, dtype=tf.float32)
        distances = tf.reshape(distances, (-1, 1))

        # separate negative and positive examples
        y_y, y_x = tf.meshgrid(tf.transpose(y_true), y_true)
        equal = tf.reshape(tf.math.equal(y_y, y_x), (-1, 1))
        n_equal = tf.math.logical_not(equal)

        positive_dist =  tf.reduce_mean(tf.boolean_mask(distances, equal))
        negative_dist =  tf.reduce_mean(tf.boolean_mask(distances, n_equal))

        return tf.maximum(positive_dist - negative_dist + alpha, .0)

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

    def training_loop(self, epochs, steps_per_epoch, data_generator, optimizer, cbc):
        loss_function = self.get_loss()
        history = {"accuracy": [], "recall": [], "loss": []}
        for epoch in range(epochs):
            for step in tqdm.tqdm(range(steps_per_epoch)):
                x, y = next(data_generator)
                with tf.GradientTape() as tape:
                    predictions = self.model(x, training=True)
                    loss = loss_function(y, predictions)

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

        callbacks_list = [lr_schedule, early_stopping, test_cb]

        self.model.run_eagerly = True
        history = self.training_loop(epochs, steps_per_epoch,
                                     self.data_generator(X_train, y_train, batch_size),
                                     optimizer, test_cb)
        return history
