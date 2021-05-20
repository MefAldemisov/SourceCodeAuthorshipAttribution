import numpy as np
import pandas as pd

from tensorflow import keras
from models.Triplet import Triplet
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from src.data_processing.commons import std_initial_preprocess


class Embedding(Triplet):

    def __init__(self,
                 input_size: int = 600,
                 output_size: int = 50,
                 make_initial_preprocess: bool = True,
                 max_val: int = 99756 + 1):

        self.max_val = max_val
        super().__init__(name="embedding",
                         input_size=input_size, output_size=output_size,
                         make_initial_preprocess=make_initial_preprocess)

    def create_model(self,
                     activation: str = "linear",
                     L2_lambda: float = 0.02,
                     conv_1_size: int = 4,
                     conv_2_size: int = 4,
                     emb_height: int = 100):

        conv_1_channels = 1
        conv_2_channels = 1
        model_core = keras.Sequential()
        model_core.add(layers.Embedding(self.max_val, emb_height,
                                        mask_zero=True, input_length=self.input_size))
        model_core.add(layers.LayerNormalization(axis=2))

        model_core.add(layers.Reshape((self.input_size, emb_height, 1)))
        model_core.add(layers.Dropout(0.5))

        model_core.add(layers.Conv2D(conv_1_channels, (conv_1_size, emb_height), padding="same", activation=activation,
                                     kernel_regularizer=regularizers.L2(L2_lambda),
                                     input_shape=(1, self.input_size, emb_height), data_format="channels_last"))

        model_core.add(layers.Conv2D(conv_2_channels, (conv_2_size, emb_height), activation=activation, padding="same",
                                     kernel_regularizer=regularizers.L2(L2_lambda),
                                     input_shape=(1, self.input_size, emb_height), data_format="channels_last"))

        model_core.add(layers.MaxPooling2D(pool_size=(self.input_size, 1), data_format="channels_last"))
        model_core.add(layers.Reshape((-1, emb_height*conv_2_channels)))

        model_core.add(layers.Flatten())

        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Dense(self.output_size))
        model_core.add(layers.LayerNormalization())

        return model_core

    @staticmethod
    def crop_to(X: np.ndarray,
                y: np.ndarray,
                crop: int = 100,
                threshold: int = 80):

        new_X = []
        new_y = []
        for old_x, old_y in zip(X, y):
            for el in old_x.reshape(-1, crop):
                if np.count_nonzero(el) > threshold:
                    new_X.append(list(el))
                    new_y.append(old_y)

        new_X = np.array(new_X).reshape((-1, crop, 1))
        new_y = np.array(new_y)
        return new_X, new_y

    def initial_preprocess(self, df_path: str, tmp_dataset_filename: str):
        std_initial_preprocess(self.input_size, df_path, tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename: str):
        dataset = pd.read_json(tmp_dataset_filename)

        X = dataset.tokens.values
        X = np.array(list(X)).reshape((-1, self.input_size))

        y = np.array(dataset.username)
        # X, y = crop_to(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
