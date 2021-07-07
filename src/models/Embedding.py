from typing import List
from src.models.base.Model import Model
from src.models.data_processing.TokenFeatures import TokenFeatures
from tensorflow.keras import layers, regularizers, models, initializers
from tensorflow import keras


class Embedding(TokenFeatures, Model):

    def __init__(self,
                 input_size: int = 100,
                 output_size: int = 50,
                 make_initial_preprocess: bool = False,
                 max_val: int = 80000 + 1):

        self.max_val = max_val
        self.output_size = output_size

        Model.__init__(self)
        TokenFeatures.__init__(self, name="embedding",
                               input_size=input_size,
                               make_initial_preprocess=make_initial_preprocess)
        self.model = self.create_model()

    def create_after_emb(self, reshape1,
                         conv_channels=1,
                         emb_height=100,
                         activation="relu",
                         L2_lambda=0.02,
                         conv_sizes=[2, 4, 16]):
        # parallel piece
        convolutions = [layers.Conv2D(conv_channels, (conv_size, emb_height),
                                      name="conv2d_size({}, {})".format(conv_size, emb_height),
                                      padding="same", activation=activation,
                                      kernel_initializer=initializers.HeNormal(),
                                      kernel_regularizer=regularizers.L2(L2_lambda),
                                      input_shape=(1, self.input_size, emb_height),
                                      data_format="channels_last")(reshape1) for conv_size in conv_sizes]

        pools = [layers.MaxPooling2D(pool_size=(self.input_size, 1),
                                     name="max_pool_size({}, {})".format(self.input_size, 1),
                                     data_format="channels_last")(conv) for conv in convolutions]

        connect = layers.concatenate(pools, axis=3)
        norm0 = layers.LayerNormalization(axis=-1)(connect)
        drop1 = layers.Dropout(0.5)(norm0)

        big_conv_channels = 2
        big_convolution = layers.Conv2D(big_conv_channels, (4, emb_height),
                                        padding="same", activation=activation,
                                        name="conv2d_size({}, {})".format(4, emb_height),
                                        kernel_initializer=initializers.HeNormal(),
                                        kernel_regularizer=regularizers.L2(L2_lambda),
                                        input_shape=(1, self.input_size, emb_height),
                                        data_format="channels_last")(drop1)  # 100, 100, 4

        reshape2 = layers.Reshape((-1, emb_height * big_conv_channels))(big_convolution)
        flatten = layers.Flatten()(reshape2)
        norm1 = layers.LayerNormalization(axis=-1)(flatten)
        drop2 = layers.Dropout(0.5)(norm1)
        dense = layers.Dense(self.output_size, activation=activation,
                             kernel_initializer=initializers.HeNormal())(drop2)
        return dense

    def create_model(self,
                     activation: str = "relu",
                     L2_lambda: float = 0.02,
                     conv_sizes: List[int] = [2, 4, 16],
                     emb_height: int = 100):

        conv_channels = 1
        input_layer = layers.Input(shape=(self.input_size, 1))

        embeddings = layers.Embedding(self.max_val, emb_height,
                                      mask_zero=True, input_length=self.input_size)(input_layer)

        reshape1 = layers.Reshape((self.input_size, emb_height, 1))(embeddings)

        dense = self.create_after_emb(reshape1, conv_channels, emb_height, activation, L2_lambda, conv_sizes)
        result = models.Model(input_layer, dense)

        print(result.summary())
        keras.utils.plot_model(result, "{}.png".format(self.name), show_shapes=True)
        return result
