from typing import List

import numpy as np
import pandas as pd

from tensorflow import keras
from models.Triplet import Triplet
from tensorflow.keras import layers, regularizers, models
from sklearn.model_selection import train_test_split
from src.data_processing.commons import std_initial_preprocess


class Embedding(Triplet):

    def __init__(self,
                 input_size: int = 100,
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
                     conv_sizes: List[int] = [2, 4, 16],
                     emb_height: int = 100):

        conv_channels = 1
        input_layer = layers.Input(shape=(self.input_size, 1))

        embeddings = layers.Embedding(self.max_val, emb_height,
                                      mask_zero=True, input_length=self.input_size)(input_layer)

        reshape1 = layers.Reshape((self.input_size, emb_height, 1))(embeddings)

        # parallel piece
        convolutions = [layers.Conv2D(conv_channels, (conv_size, emb_height),
                                      padding="same", activation=activation,
                                      kernel_regularizer=regularizers.L2(L2_lambda),
                                      input_shape=(1, self.input_size, emb_height),
                                      data_format="channels_last")(reshape1) for conv_size in conv_sizes]


        pools = [layers.MaxPooling2D(pool_size=(self.input_size, 1),
                                     data_format="channels_last")(conv) for conv in convolutions]

        connect = layers.concatenate(pools, axis=3)
        norm0 = layers.LayerNormalization(axis=-1)(connect)
        drop1 = layers.Dropout(0.5)(norm0)

        big_conv_channels = 2
        big_convolution = layers.Conv2D(big_conv_channels, (4, emb_height),
                                        padding="same", activation=activation,
                                        kernel_regularizer=regularizers.L2(L2_lambda),
                                        input_shape=(1, self.input_size, emb_height),
                                        data_format="channels_last")(drop1) # 100, 100, 4

        reshape2 = layers.Reshape((-1, emb_height * big_conv_channels))(big_convolution)
        flatten = layers.Flatten()(reshape2)
        norm1 = layers.LayerNormalization(axis=-1)(flatten)
        drop2 = layers.Dropout(0.5)(norm1)
        dense = layers.Dense(self.output_size)(drop2)
        result =  models.Model(input_layer, dense)

        print(result.summary())
        keras.utils.plot_model(result, "{}.png".format(self.name), show_shapes=True)

        return result

    @staticmethod
    def crop_to(X: np.ndarray,
                y: np.ndarray,
                crop: int = 100,
                threshold: int = 80):

        new_X = []
        new_y = []
        for old_x, old_y in zip(X, y):
            for el in old_x.reshape(-1, crop):
                if np.count_nonzero(el) > threshold * crop // 100:
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
        X, y = self.crop_to(X, y, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
