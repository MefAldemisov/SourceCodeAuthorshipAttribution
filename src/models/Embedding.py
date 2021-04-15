import numpy as np
import pandas as pd

from models.Triplet import Triplet
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from src.data_processing.commons import std_initial_preprocess


class Embedding(Triplet):

    def __init__(self, triplet_type="default", input_size=600, output_size=50,
                 make_initial_preprocess=True, max_val=99756+1):
        self.max_val = max_val
        super().__init__(name="embedding",
                         input_size=input_size, output_size=output_size,
                         make_initial_preprocess=make_initial_preprocess, triplet_type=triplet_type)

    def create_model(self, activation="linear", L2_lambda=0.02,
                     conv_1_size=64, conv_2_size=16, emb_height=100):

        model_core = keras.Sequential()
        model_core.add(layers.Embedding(self.max_val, emb_height,
                                        mask_zero=True, input_length=self.input_size))

        model_core.add(layers.Reshape((self.input_size, emb_height, 1)))
        model_core.add(layers.Dropout(0.5))

        model_core.add(layers.Conv2D(16, (conv_1_size, emb_height), padding="same", activation=activation,
                                     kernel_regularizer=regularizers.L2(L2_lambda),
                                     input_shape=(1, self.input_size, emb_height), data_format="channels_last"))

        model_core.add(layers.Conv2D(16, (conv_2_size, emb_height), activation=activation, padding="same",
                                     kernel_regularizer=regularizers.L2(L2_lambda),
                                     input_shape=(1, self.input_size, emb_height), data_format="channels_last"))

        model_core.add(layers.MaxPooling2D(pool_size=(self.input_size, 1), data_format="channels_last"))
        model_core.add(layers.Reshape((-1, emb_height*16)))

        model_core.add(layers.Flatten())

        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Dense(self.output_size))
        return model_core

    @staticmethod
    def crop_to(X, y, crop=100, threshold=80):
        new_X = []
        new_y = []
        for old_x, old_y in zip(X, y):
            for el in old_x.reshape(-1, crop):
                if np.count_nonzero(el) > threshold:
                    new_X.append(list(el))
                    new_y.append(old_y)

        new_X = np.array(new_X).reshape(-1, crop, 1)
        new_y = np.array(new_y)
        return new_X, new_y

    def initial_preprocess(self, df_path, tmp_dataset_filename):
        std_initial_preprocess(self.input_size, df_path, tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename):
        dataset = pd.read_json(tmp_dataset_filename)

        X = dataset.tokens.values
        X = np.array(list(X)).reshape((-1, self.input_size))

        y = np.array(dataset.username)
        # X, y = crop_to(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test

