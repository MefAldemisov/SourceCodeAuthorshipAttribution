import torch
import numpy as np

'''
The loader of the train data (batch generator)
'''


class GCJ:

    def __init__(self, X_train, y_train, batch_size, input_size):
        self.x = X_train
        self.y = y_train
        self.batch_size = batch_size
        self.input_size = input_size

    def batch_generator(self, model, tree):
        n_positive = self.batch_size // 2
        anchor_index = np.random.choice(self.y.shape[0], 1)
        y_anchor = self.y[anchor_index]
        positive_indexes = np.where(self.y == y_anchor)[0]
        n_same = positive_indexes.shape[0]
        positive_indexes = positive_indexes[:n_positive]
        k = self.batch_size - positive_indexes.shape[0]

        if tree is not None:
            with torch.no_grad():
                query = model(self.x[anchor_index])
            query_res = tree.query(query, self.batch_size+n_same, return_distance=False)[0]
            negative_indexes = np.array([neighbour_index for neighbour_index in query_res
                                         if self.y[neighbour_index] != y_anchor])[:k]
        else:  # the first batch generation
            negative_indexes = np.where(self.y != y_anchor)[0]
            np.random.shuffle(negative_indexes)
            negative_indexes = negative_indexes[:k]

        local_x = self.x.reshape((-1, self.input_size, 768))

        reduced_indexes = map(lambda indexes: np.random.choice(indexes, self.batch_size),
                              [positive_indexes, negative_indexes])

        positive, negative = map(lambda i: local_x[i], reduced_indexes)
        anchor = torch.concat([local_x[anchor_index] for _ in range(self.batch_size)])

        return anchor, positive, negative
