import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from tensorflow.keras import callbacks
from sklearn.metrics import accuracy_score, confusion_matrix


class TestCallback(callbacks.Callback):

    def __init__(self, X_train, X_test, y_train, y_test, threshold: float = 0.1,
                 input_size: int = 500, n_authors: int = 20, model_name: str = "conv2d"):
        """
        Parameters:
        - `X_train`,` X_test` - np.arrays with data (tokens)
        - `y_train`, `y_test` - np.arrays, labels (numerical representation of authors)

        -  `threshold` - alpha parameter of the triplet loss, threshold for the classification's distance
        -  `input_size` - amount of tokens in one file
        -  `n_authors` - int, prediction stage requires the all-with-all comparison (O(n^2)),
        that is why, it is reduced for plotting and evaluating
        """
        super().__init__()
        self.threshold = threshold
        self.input_size = input_size
        # creation of summary-writer
        current_time = datetime.datetime.now().strftime("%Y>%m>%d-%H>%M>%S")
        test_log_dir = "../outputs/tensor_board/{}/test".format(current_time)
        train_log_dir = "../outputs/tensor_board/{}/train".format(current_time)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # x-y preprocessing
        self.n_authors = n_authors

        index = np.where(y_train < self.n_authors)[0]
        self.X_train = X_train[index]
        self.y_train = y_train[index]

        index = np.where(y_test < self.n_authors)[0]
        self.X_test = X_test[index]
        self.y_test = y_test[index]
        # counter initialization
        self.n = 0
        self.model_name = model_name

    def apply_dimensionality_reduction(self, transformed_x, y):
        vectors = TSNE(n_components=2)
        x_pca = vectors.fit_transform(transformed_x)
        plt.figure(figsize=(10, 8))
        plt.title("Step {} (epoch {})".format(self.n, self.n//125))
        for developer in range(self.n_authors):
            indexes = np.where(y == developer)[0]
            plt.plot(x_pca[indexes, 0], x_pca[indexes, 1], "o", ms=5)
        plt.savefig("../outputs/tsne_{}/tsne_{}.png".format(self.model_name, self.n))
        plt.close('all')
        self.n += 1

    def get_acc_and_recall(self, model, x, y, plot: bool = False):
        transformed_x = model.predict(x.reshape(-1, self.input_size))

        mse = lambda a, b: np.mean((a - b) ** 2)
        y_pred, y_true = [], []
        for i in range(x.shape[0]):
            for j in range(i, x.shape[0]):
                distance = mse(transformed_x[i], transformed_x[j])
                y_pred.append(int(distance <= self.threshold))
                y_true.append(int(y[i] == y[j]))

        y_pred, y_true = np.array(y_pred), np.array(y_true)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        recall = cm[1][1] / sum(y_true)
        if plot:
            self.apply_dimensionality_reduction(transformed_x, y)
        return accuracy, recall

    def on_epoch_end(self, model, epoch, loss, logs=None):
        test_accuracy, test_recall = self.get_acc_and_recall(model, self.X_test, self.y_test, plot=True)
        with self.test_summary_writer.as_default():
            tf.summary.scalar("test_accuracy", test_accuracy, step=self.n)
            tf.summary.scalar("test_recall", test_recall, step=self.n)

        train_accuracy, train_recall = self.get_acc_and_recall(model, self.X_train, self.y_train, plot=False)
        with self.train_summary_writer.as_default():
            tf.summary.scalar("train_accuracy", train_accuracy, step=self.n)
            tf.summary.scalar("train_recall", train_recall, step=self.n)
            tf.summary.scalar("train_loss", loss, step=self.n)
