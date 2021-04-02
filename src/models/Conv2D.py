import numpy as np
import pandas as pd

from models.Triplet import Triplet
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class Conv2D(Triplet):
    def __init__(self, output_size=50, img_x=120, img_y=200, make_initial_preprocess=True):
        self.img_x, self.img_y = img_x, img_y
        super().__init__("conv2d", input_size=img_x*img_y, output_size=output_size,
                         make_initial_preprocess=make_initial_preprocess)

    def create_model(self):
        model_core = keras.Sequential()
        model_core.add(layers.Reshape((self.img_y, self.img_x, 1)))
        model_core.add(layers.Conv2D(16, 8, activation="tanh", padding="same"))
        model_core.add(layers.MaxPooling2D(pool_size=4))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Conv2D(16, 4, activation="tanh", padding="same"))
        model_core.add(layers.MaxPooling2D(pool_size=4))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Conv2D(16, 4, activation="tanh", padding="same"))
        model_core.add(layers.MaxPooling2D(pool_size=4))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Flatten())
        model_core.add(layers.Dense(128, activation="tanh"))
        model_core.add(layers.LayerNormalization(axis=1))
        model_core.add(layers.Dropout(0.5))
        model_core.add(layers.Dense(self.output_size, activation="tanh"))
        return model_core

    def initial_preprocess(self, df_path, tmp_dataset_filename):
        df = pd.read_csv(df_path)
        df = df.drop(columns=["round", "task", "solution", "file", "full_path", "Unnamed: 0.1", "Unnamed: 0", "lang"])
        df["n_lines"] = df.flines.apply(lambda x: str(x).count("\n"))
        df["n_lines"].describe()
        df = df[(df.n_lines >= 30) & (df.n_lines < self.img_y)]  # there should be enough loc for 2D convolution

        def max_cpl(file):
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
        users = df.username.value_counts()[0:650].index
        df = df[df.username.isin(users)]
        # string to int for y
        le = LabelEncoder()
        df.username = le.fit_transform(df.username)
        # save
        dataset = df[["username", "flines"]]
        dataset.to_json(tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename):
        df = pd.read_json(tmp_dataset_filename)

        def vectorize(file):
            lines = file.split('\n')
            res = np.zeros((self.img_y, self.img_x), dtype=int)
            for i in range(len(lines)):
                if i >= self.img_y:
                    break
                line = lines[i]
                for j in range(len(line)):
                    res[i][j] = ord(line[j])
            return res.tolist()

        X = df.flines.apply(vectorize).values
        X = np.array([np.array(x) for x in X])
        X = X.reshape(-1, self.img_y, self.img_x)

        y = np.array(df.username)

        ss = StandardScaler()
        X = ss.fit_transform(X.reshape((-1, self.img_y * self.img_x))).reshape((-1, self.img_y, self.img_x))

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
