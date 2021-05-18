import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from tensorflow.keras import callbacks
from sklearn.metrics import accuracy_score, confusion_matrix


class TestCallback(callbacks.Callback):

    def __init__(self, X, y, empty_model=None, threshold=0.1, input_size=500):
        """
        Parameters:
        - `X, y` - np.arrays with data (tokens) and labels (numerical representation of authors)
        - `empty_model` - tensorflow.keras.model object, which will be used to copy the weights if the 'default'
        type of triplet loss is used
        -  `threshold` - alpha parameter of the triplet loss, threshold for the classification's distance
        -  `input_size` - amount of tokens in one file
        -  `is_default` - bool, if true, then it is necessary to generate the copy of the `self.model.weights`
        to the `empty_model` model (due to the triplet nature of the model)
        """
        super().__init__()
        self.threshold = threshold
        self.input_size = input_size
        self.n_authors = 20
        index = np.where(y < self.n_authors)[0]
        self.X = X[index]
        self.y = y[index]
        self.scores = []
        self.recalls = []
        self.local_model = empty_model
        self.local_model.build((None, self.input_size))
        self.n = 0

    def apply_pca(self, transformed_x):
        pca = PCA(n_components=3)
        x_pca = pca.fit_transform(transformed_x)
        plt.figure(figsize=(10, 8))
        for developer in range(self.n_authors):
            indexes = np.where(self.y == developer)
            plt.plot(x_pca[indexes][0], x_pca[indexes][1], "o", ms=5)
        plt.savefig("../outputs/pca/pca_{}.png".format(self.n))
        plt.close('all')
        self.n += 1

    def on_epoch_end(self, model, epoch, logs=None):

        transformed_x = model.predict(self.X.reshape(-1, self.input_size))

        y_pred = []
        y_true = []
        for i in range(self.X.shape[0]):
            for j in range(i, self.X.shape[0]):
                if np.mean((transformed_x[i] - transformed_x[j]) ** 2) <= self.threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

                if self.y[i] == self.y[j]:
                    y_true.append(1)
                else:
                    y_true.append(0)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        score = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        recall = cm[1][1] / sum(y_true)

        self.scores.append(score)
        self.recalls.append(recall)
        print("accuracy:", round(score, 4), "recall:", round(recall, 4))
        self.apply_pca(transformed_x)
        return score, recall

    def return_results(self):
        return {"scores": self.scores, "recalls": self.recalls}
