import numpy as np
import pandas as pd

from tensorflow import keras
from models.Triplet import Triplet
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Conv2D(Triplet):
    def __init__(self,
                 output_size: int = 50,
                 img_x: int = 120,
                 img_y: int = 200,
                 make_initial_preprocess: bool = True):

        self.img_x, self.img_y = img_x, img_y
        super().__init__("conv2d", input_size=img_x*img_y, output_size=output_size,
                         make_initial_preprocess=make_initial_preprocess)

    def create_model(self):
        emb_height = 100
        model_core = keras.Sequential()
        model_core.add(layers.Input((self.img_y, self.img_x)))
        model_core.add(layers.Reshape((self.img_y * self.img_x, 1)))
        model_core.add(layers.Embedding(756452 + 1, emb_height, mask_zero=True,
                                        input_length=self.img_y * self.img_x))  # output shape: (-1, x*y, 100)

        # model_core.add(layers.LayerNormalization(axis=2))
        model_core.add(layers.Reshape((self.img_y * self.img_x, 100, 1)))

        # pooling to reduce the dimensionality:
        model_core.add(layers.AveragePooling2D(pool_size=(1, 100),
                                               ata_format="channels_last"))  # output shape: -1, 1, x*y, 1
        model_core.add(layers.Reshape((self.img_y, self.img_x, 1)))

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

    @staticmethod
    def _path_to_x(y_path: str):
        x_path = y_path[:-4]  # "json" removed
        x_path += "txt"
        return x_path

    def initial_preprocess(self,
                           df_path: str,
                           tmp_dataset_filename: str):

        df = pd.read_csv(df_path)
        df = df.drop(columns=["round", "task", "solution", "file", "full_path", "Unnamed: 0.1", "Unnamed: 0", "lang"])
        df["n_lines"] = df.flines.apply(lambda x: str(x).count("\n"))
        df = df[(df.n_lines >= 30) & (df.n_lines < self.img_y)]  # there should be enough loc for 2D convolution

        def max_cpl(file: str):
            """"
            Max chars per line
            """
            lines = file.split('\n')
            max_ch = 0
            for line in lines:
                if len(line) > max_ch:
                    max_ch = len(line)
            return max_ch

        df["max_cpl"] = df.flines.apply(max_cpl)
        df = df[df.max_cpl <= self.img_x]
        # select users
        users = df.username.value_counts()[0:500].index
        df = df[df.username.isin(users)]
        # string to int for y
        le = LabelEncoder()
        df.username = le.fit_transform(df.username)

        def to_vector(file: str):
            lines = file.split('\n')
            res = np.zeros((self.img_y, self.img_x), dtype=int)
            for i in range(len(lines)):
                if i >= self.img_y:
                    break
                line = lines[i]
                for j in range(len(line)):
                    res[i][j] = ord(line[j])
            return res.tolist()

        X = df.flines.apply(to_vector).values
        X = np.array([np.array(x) for x in X])
        # scale
        # ss = StandardScaler()
        # X = ss.fit_transform(X.reshape((-1, self.img_y * self.img_x))).reshape((-1, self.img_y, self.img_x))
        X = X.reshape((-1, self.img_y, self.img_x))
        # save X and y separately
        x_file = open(self._path_to_x(tmp_dataset_filename), "wb")
        np.save(x_file, X)
        x_file.close()
        dataset = df[["username"]]
        dataset.to_json(tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename: str):
        df = pd.read_json(tmp_dataset_filename)
        y = np.array(df.username)
        # read X
        file = open(self._path_to_x(tmp_dataset_filename), "rb")
        X = np.load(file)
        file.close()
        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
