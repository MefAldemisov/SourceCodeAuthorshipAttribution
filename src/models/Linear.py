from src.models.base.Model import Model
from src.models.data_processing.TokenFeatures import TokenFeatures

from tensorflow import keras
from tensorflow.keras import layers, regularizers


class Linear(TokenFeatures, Model):

    def __init__(self,
                 input_size: int = 600,
                 output_size: int = 50,
                 make_initial_preprocess: bool = True):
        # name left the same, because training data and its preprocessing are the same
        # as for 'Embedding(Triplet)'

        self.output_size = output_size
        Model.__init__()
        TokenFeatures.__init__(name="embedding",
                               input_size=input_size,
                               make_initial_preprocess=make_initial_preprocess)

    def create_model(self,
                     activation: str = "linear",
                     L2_lambda: float = 0.02,
                     pool_1_size: int = 4,
                     pool_2_size: int = 4,
                     conv_1_size: int = 16,
                     conv_2_size: int = 4,
                     dense_1: int = 64):

        model_core = keras.Sequential()
        model_core.add(layers.Reshape((-1, 1)))
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
