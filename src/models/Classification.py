import numpy as np
import pandas as pd

from models.Model import *
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from src.data_processing.commons import std_initial_preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Classification(Model):
    def __init__(self, input_length=600, n_classes=40, make_initial_preprocess=True):
        super().__init__(name="classification", make_initial_preprocess=make_initial_preprocess)
        self.input_length = input_length
        self.n_classes = n_classes
        self.model = self.create_model()

    def initial_preprocess(self, df_path, tmp_dataset_filename):
        std_initial_preprocess(self.input_length, df_path, tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename, scale=False):
        dataset = pd.read_json(tmp_dataset_filename)
        X = np.array(list(dataset.tokens.values)).reshape((-1, self.input_length, 1))
        y = np.array(dataset.username)

        X = X[np.where(y < self.n_classes)]
        y = y[np.where(y < self.n_classes)]

        ohe = OneHotEncoder()
        y = ohe.fit_transform(y.reshape(-1, 1)).toarray()

        if scale:
            ss = StandardScaler()
            X = X.reshape((-1, self.input_length))
            ss.fit(X)
            X = ss.transform(X)
            X = X.reshape((-1, self.input_length, 1))

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test

    def create_model(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.input_length, 1)))
        model.add(layers.Conv1D(150, 10, activation="linear"))
        model.add(layers.MaxPooling1D(pool_size=8))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation="linear"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.n_classes, activation="softmax"))
        # second arch
        # model.add(layers.Conv1D(64, 16, activation="tanh"))
        # model.add(layers.MaxPooling1D(pool_size=8))
        # model.add(layers.Conv1D(32, 4, activation="linear"))
        # model.add(layers.MaxPooling1D(pool_size=8))
        # model.add(layers.Flatten())
        # model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(100, activation="linear"))
        # model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(40, activation="softmax"))
        return model

    def train(self, epochs, batch_size):
        X_train, X_test, y_train, y_test = self.preprocess()

        optimizer = optimizers.Adam(0.1)
        lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5,
                                                  min_delta=0.000001, verbouse=1)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        history = self.model.fit(x=X_train, y=y_train, epochs=epochs,
                                 validation_data=(X_test, y_test),
                                 steps_per_epoch=X_train.shape[0] // batch_size,
                                 verbose=2, callbacks=[lr_schedule])
        return history
