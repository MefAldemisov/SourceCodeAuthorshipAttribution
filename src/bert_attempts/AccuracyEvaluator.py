import io
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import List
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class AccuracyEvaluator:

    def __init__(self,
                 # X_train: np.ndarray,
                 X_test: np.ndarray,
                 # y_train: np.ndarray,
                 y_test: np.ndarray,
                 threshold: float = 0.1,
                 input_size: int = 500,
                 authors: List = list(range(20))):
        """
        Parameters:
        - `X_train`,` X_test` - np.arrays with data (tokens)
        - `y_train`, `y_test` - np.arrays, labels (numerical representation of authors)

        -  `threshold` - alpha parameter of the triplet loss, threshold for the classification's distance
        -  `input_size` - amount of tokens in one file
        -  `authors` - int, prediction stage requires the all-with-all comparison (O(n^2)),
        that is why, it is reduced for plotting and evaluating
        """
        super().__init__()
        self.threshold = threshold
        self.input_size = input_size
        # x-y preprocessing
        self.authors = authors

        def select_authors(initial_x, initial_y):
            index = np.where(np.isin(initial_y, self.authors))[0]
            new_x, new_y = map(lambda a: a[index], [initial_x, initial_y])
            return new_x, new_y

        # simple_x_train, simple_y_train = select_authors(X_train, y_train)
        simple_x_test, simple_y_test = select_authors(X_test, y_test)

        self.data = {
            "simple": {
                # "train": [simple_x_train, simple_y_train],
                "test": [simple_x_test, simple_y_test]
            },
            "full": {
                # "train": [X_train, y_train],
                "test": [X_test, y_test]
            }
        }

        # counter initialization
        self.n = 0

    @staticmethod
    def _plot_to_image(figure):
        # https://www.tensorflow.org/tensorboard/image_summaries
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(figure)
        buf.seek(0)

    def apply_dimensionality_reduction(self,
                                       transformed_x: np.ndarray,
                                       y: np.ndarray,
                                       epoch: int,
                                       is_test: bool):
        vectors = TSNE(n_components=2)
        x_pca = vectors.fit_transform(transformed_x)
        figure = plt.figure(figsize=(10, 8))
        plt.title("Step {} (epoch {})".format(self.n, epoch))
        for developer in self.authors:
            indexes = np.where(y == developer)[0]
            plt.plot(x_pca[indexes, 0], x_pca[indexes, 1], "o", ms=5)
        # save as file
        plt.savefig("../outputs/tsne_{}/tsne_{}.png".format('bert', self.n))
        # log to tensorboard
        # image = self._plot_to_image(figure)
        # writer = self.test_summary_writer if is_test else self.train_summary_writer
        # with writer.as_default():
        #     tf.summary.image("Distribution of authors", image, step=self.n)

        plt.close("all")

    def get_acc(self,
                model,
                x: np.ndarray,
                y: np.ndarray,
                epoch: int,
                is_test: bool,
                dim_red: True) -> float:

        transformed_x = model(x)
        knn = KNeighborsClassifier().fit(transformed_x, y)
        predictions = knn.predict(transformed_x)
        accuracy = accuracy_score(y_true=y, y_pred=predictions)
        # if dim_red:
        #     self.apply_dimensionality_reduction(transformed_x, y, epoch, is_test)
        return accuracy

    def _writer(self,
                x,
                y,
                model,
                epoch: int,
                is_test: bool,
                is_simple: bool) -> float:

        accuracy = self.get_acc(model, x, y, epoch, is_test, is_simple)
        return accuracy

    def on_epoch_end(self,
                     model,
                     epoch: int,
                     loss: float):

        # astr = self._writer(*self.data["simple"]["train"], model, epoch, False, True)
        aste = self._writer(*self.data["simple"]["test"], model, epoch, True, True)
        afte = self._writer(*self.data["full"]["test"], model, epoch, True, False)

        print(loss, aste, afte)
        self.n += 1
        return aste, afte
