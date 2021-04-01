import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import callbacks


class TestCallback(callbacks.Callback):

    def __init__(self, X, y, emb_model, threshold=0.2, input_size=500):
        super().__init__()
        self.threshold = threshold
        self.input_size = input_size
        self.emb_model = emb_model
        index = np.where(y < 5)[0]
        self.X = X[index]
        self.y = y[index]
        self.scores = []
        self.recalls = []

    def on_epoch_end(self, epoch, logs=None):
        # (model, new_X, new_y, threshold=0.2, printing=0):
        transformed_X = self.emb_model.predict(self.X.reshape(-1, self.input_size, 1))
        #     X_train, X_test, y_train, y_test = train_test_split(transformed_X, new_y)

        y_pred = []
        y_true = []
        for i in range(self.X.shape[0]):
            for j in range(i, self.X.shape[0]):
                if np.mean((transformed_X[i] - transformed_X[j]) ** 2) <= self.threshold:
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
        #         if printing:
        #             print(cm)
        #             print("total predicted as 1:", sum(y_pred),
        #                   "\n total real 1:", sum(y_true),
        #                   "\n total examples", y_true.shape)

        recall = cm[1][1] / sum(y_true)

        self.scores.append(score)
        self.recalls.append(recall)
        print("accuracy:", round(score, 4), "recall:", round(recall, 4))
        return score, recall

    def return_results(self):
        return {"scores": self.scores, "recalls": self.recalls}
