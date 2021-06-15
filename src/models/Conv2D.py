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
                 make_initial_preprocess: bool = True):

        self.output_size = output_size  # TODO: move to Model, probably, input_size -- ??
        Model.__init__()
        CharFeatures.__init__(name="conv2d",
                              img_x=img_x, img_y=img_y, crop=crop,
                              make_initial_preprocess=make_initial_preprocess)
        self.model = self.create_model()

    def create_model(self):
        emb_height = 100
        model_core = keras.Sequential()
        model_core.add(layers.Input((self.crop, self.img_x)))
        model_core.add(layers.Reshape((self.crop * self.img_x, 1)))
        model_core.add(layers.Embedding(756452 + 1, emb_height, mask_zero=True,
                                        input_length=self.crop * self.img_x))  # output shape: (-1, x*y, 100)

        # model_core.add(layers.LayerNormalization(axis=2))
        model_core.add(layers.Reshape((self.crop * self.img_x, 100, 1)))

        # pooling to reduce the dimensionality:
        model_core.add(layers.AveragePooling2D(pool_size=(1, 100),
                                               data_format="channels_last"))  # output shape: -1, 1, x*y, 1
        model_core.add(layers.Reshape((self.crop, self.img_x, 1)))

        model_core.add(layers.Conv2D(16, 16, activation="relu", padding="same"))
        model_core.add(layers.MaxPooling2D(pool_size=4))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Conv2D(16, 4, activation="relu", padding="same"))
        model_core.add(layers.MaxPooling2D(pool_size=4))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Conv2D(16, 4, activation="relu", padding="same"))
        model_core.add(layers.MaxPooling2D(pool_size=4))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Flatten())
        model_core.add(layers.Dense(128, activation="tanh"))
        model_core.add(layers.LayerNormalization(axis=1))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Dense(self.output_size, activation="tanh"))
        return model_core
