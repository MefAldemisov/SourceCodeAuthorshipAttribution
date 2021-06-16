from src.models.base.Model import Model
from src.models.data_processing.CharFeatures import CharFeatures

from tensorflow import keras
from tensorflow.keras import layers, regularizers


class Conv2D(CharFeatures, Model):
    def __init__(self,
                 output_size: int = 50,
                 img_x: int = 120,
                 img_y: int = 200,
                 crop=None,
                 make_initial_preprocess: bool = False):

        self.output_size = output_size  # TODO: move to Model, probably, input_size -- ??
        Model.__init__(self)
        CharFeatures.__init__(self, name="conv2d",
                              img_x=img_x, img_y=img_y, crop=crop,
                              make_initial_preprocess=make_initial_preprocess)
        self.input_size = img_x * self.crop
        self.model = self.create_model()

    def create_after_emb(self, reshape1,
                         conv_channels=2,
                         emb_height=100,
                         activation="relu",
                         L2_lambda=0.02,
                         conv_sizes=[4, 8, 16]):
        conv3d = layers.Conv3D(1, (4, 4, 10), padding="same",
                               activation=activation, kernel_regularizer=regularizers.L2(L2_lambda),
                               data_format="channels_last")(reshape1)
        pooling3d = layers.MaxPooling3D(pool_size=(1, 1, emb_height), data_format="channels_last")(conv3d)
        rs = layers.Reshape((self.crop, self.img_x, 1))(pooling3d)
        # parallel piece
        convolutions = [layers.Conv2D(conv_channels, (conv_size, conv_size),
                                      padding="same", activation=activation,
                                      kernel_regularizer=regularizers.L2(L2_lambda),
                                      data_format="channels_last")(rs) for conv_size in conv_sizes]

        pools = [layers.MaxPooling2D(pool_size=4, padding="same",
                                     data_format="channels_last")(conv) for conv in convolutions]

        connect = layers.concatenate(pools, axis=3)
        norm0 = layers.LayerNormalization(axis=-1)(connect)
        drop1 = layers.Dropout(0.5)(norm0)

        big_conv_channels = 1
        big_convolution = layers.Conv2D(big_conv_channels, (4, emb_height),
                                        padding="same", activation=activation,
                                        kernel_regularizer=regularizers.L2(L2_lambda),
                                        data_format="channels_last")(drop1)  # 100, 100, 4

        flatten = layers.Flatten()(big_convolution)
        norm1 = layers.LayerNormalization(axis=-1)(flatten)
        drop2 = layers.Dropout(0.5)(norm1)
        dense = layers.Dense(self.output_size)(drop2)
        return dense

    def create_model(self):
        emb_height = 100

        input_layer = layers.Input((self.crop * self.img_x, 1))
        # rs1 = layers.Reshape((self.crop * self.img_x, 1))(input_layer)
        embedding = layers.Embedding(756452 + 1, emb_height, mask_zero=True,
                                     input_length=self.crop * self.img_x)(input_layer)
        rs2 = layers.Reshape((self.crop, self.img_x, emb_height, 1))(embedding)

        # parallelism
        dense = self.create_after_emb(rs2, conv_channels=1)

        result = keras.models.Model(input_layer, dense)
        keras.utils.plot_model(result, "{}.png".format(self.name), show_shapes=True)
        return result
