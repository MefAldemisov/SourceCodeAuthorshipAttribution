import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers, regularizers, models


df = pd.read_csv("../inputs/processed_dfs/training.1600000.processed.noemoticon.csv")
df.columns = ["label", "id", "date", "query", "nickname", "text"]
df = df[["label", "text"]]
df.label[df.label == 4] = 1

# the dataset is too big
df = df[700000:900000]


# convert text_length to char-by-char representation
def to_vector(file: str, x_=120, y_=3):
    res = np.zeros((y_, x_), dtype=int)
    for i in range(len(file)):
        assert i // x_ <= y_ and i % x_ < x_, "invalid i {}".format(i)
        res[i // x_][i % x_] = ord(file[i])
    return res.tolist()


df["x"] = df.text.apply(to_vector)
X, y = np.array(list(df.x.values)), np.array(list(df.label.values))

X = np.array(X).reshape(-1, 3, 120)
y = np.array(y).reshape(-1, 1)
y = OneHotEncoder().fit_transform(y).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y)


def create_model(conv_channels=1,
                 emb_height=100,
                 activation="relu",
                 L2_lambda=0.02,
                 conv_sizes=[4, 8, 16]):
    input_layer = layers.Input((360, 1))
    # rs1 = layers.Reshape((self.crop * self.img_x, 1))(input_layer)
    embedding = layers.Embedding(65533 + 1, emb_height, mask_zero=True,
                                 input_length=360)(input_layer)
    rs2 = layers.Reshape((3, 120, emb_height, 1))(embedding)
    conv3d = layers.Conv3D(1, (4, 4, 10), padding="same",
                           activation=activation, kernel_regularizer=regularizers.L2(L2_lambda),
                           data_format="channels_last")(rs2)
    pooling3d = layers.MaxPooling3D(pool_size=(1, 1, emb_height), data_format="channels_last")(conv3d)
    rs = layers.Reshape((3, 120, 1))(pooling3d)
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
    dense = layers.Dense(50, activation=activation)(drop2)
    pred = layers.Dense(2, activation="softmax")(dense)
    result = models.Model(input_layer, pred)
    return result


model = create_model()

model.compile(optimizer="adam",
              metrics=["accuracy"],
              loss="categorical_crossentropy")

history = model.fit(X_train.reshape(-1, 360, 1), y_train, batch_size=128,
                    validation_data=(X_test.reshape(-1, 360, 1), y_test))
