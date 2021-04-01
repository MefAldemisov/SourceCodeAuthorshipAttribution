import numpy as np
import pandas as pd

from models.Triplet import Triplet
from tensorflow import keras
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from src.data_processing.commons import std_initial_preprocess


class Linear(Triplet):

    def __init__(self, input_size=600, output_size=50, make_initial_preprocess=True):
        # name left the same, because training data and its preprocessing are the same
        # as for 'Embedding(Triplet)'

        super().__init__("embedding", input_size, output_size,
                         make_initial_preprocess=make_initial_preprocess)

    def create_model(self, activation="linear", L2_lambda=0.02,
                     pool_1_size=4, pool_2_size=4,
                     conv_1_size=16, conv_2_size=4, dense_1=64):
        model_core = keras.Sequential()
        model_core.add(layers.Conv1D(64, conv_1_size,
                                     activation=activation,
                                     kernel_regularizer=regularizers.L2(L2_lambda)))

        model_core.add(layers.LayerNormalization(axis=1))
        model_core.add(layers.MaxPooling1D(pool_size=pool_1_size))

        model_core.add(layers.Conv1D(32, conv_2_size,
                                     activation=activation,
                                     kernel_regularizer=regularizers.L2(L2_lambda)))

        model_core.add(layers.LayerNormalization(axis=1))
        model_core.add(layers.MaxPooling1D(pool_size=pool_2_size))

        model_core.add(layers.Flatten())
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Dense(dense_1, activation=activation,
                                    kernel_regularizer=regularizers.L2(L2_lambda)))
        model_core.add(layers.LayerNormalization(axis=1))

        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Dense(self.output_size, activation=activation,
                                    kernel_regularizer=regularizers.L2(L2_lambda)))
        model_core.add(layers.LayerNormalization(axis=1))
        return model_core

    def create_triplet_model(self):
        input_anchor = layers.Input(shape=(self.input_size, 1))
        input_positive = layers.Input(shape=(self.input_size, 1))
        input_negative = layers.Input(shape=(self.input_size, 1))

        model_anchor = self.model(input_anchor)
        model_positive = self.model(input_positive)
        model_negative = self.model(input_negative)

        result = layers.concatenate([model_anchor, model_positive,
                                     model_negative], axis=1)
        triplet = models.Model(
            [input_anchor, input_positive, input_negative], result)
        return triplet

    def initial_preprocess(self, df_path, tmp_dataset_filename):
        std_initial_preprocess(self.input_size, df_path, tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename):
        dataset = pd.read_json(tmp_dataset_filename)

        X = dataset.tokens.values
        X = np.array(list(X)).reshape((-1, self.input_size, 1))

        y = np.array(dataset.username)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
