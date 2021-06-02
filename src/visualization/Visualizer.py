import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from src.models.Triplet import Triplet
from tf_keras_vis.scorecam import ScoreCAM

# https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb


class Visualizer(Triplet):
    def __init__(self, model_name: str):
        super().__init__(name="none", make_initial_preprocess=False)
        self.model_name = model_name
        self.model = tf.keras.models.load_model('../outputs/{}.h'.format(model_name))
        all_x, all_y = self.read_conv2d_dataset() if model_name == "conv2d" else self.read_embedding_dataset()
        self.x_batch, self.y_batch = self._batches_generator(all_x, all_y, 128)
        self.x_author = self.x_batch[0]

    @staticmethod
    def create_model():
        return None

    def loss(self, output: tf.Tensor):
        l = self.hard_triplet_loss(self.y_batch, output)
        return [l for _ in output]

    @staticmethod
    def model_modifier(m: tf.keras.Model):
        new_model = Sequential()
        new_model.add(Input((600, 100, 1)))
        for layer in m.layers[3:]:
            new_model.add(layer)
        return new_model

    def read_conv2d_dataset(self):
        # load dataset
        df = pd.read_json("../inputs/preprocessed_jsons/{}_train.json".format(self.model_name))
        y = np.array(df.username)
        # read X
        file = open("../inputs/preprocessed_jsons/{}_train.txt".format(self.model_name), "rb")
        x = np.load(file)
        file.close()
        return x, y

    def read_embedding_dataset(self):
        dataset = pd.read_json("../inputs/preprocessed_jsons/{}_train.json".format(self.model_name))
        X = dataset.tokens.values
        X = np.array(list(X)).reshape((-1, 600))

        y = np.array(dataset.username)
        return X, y

    def run_conv2d(self):
        saliency = Gradcam(self.model, model_modifier=self.model_modifier, clone=False)

        x_batch = tf.reshape(self.x_batch, (-1, 200, 120, 1))
        # saliency_map = saliency(self.loss, x_batch, smooth_samples=20, smooth_noise=0.2)
        saliency_map = saliency(self.loss, x_batch, penultimate_layer=-1)
        heatmap = saliency_map[0]
        # plotting
        plt.figure(figsize=(18, 6))
        plt.subplot(121)
        plt.title("Source code")
        plt.imshow(self.x_author)
        plt.subplot(122)
        plt.title("GradCAM map")
        plt.imshow(heatmap)

    def run_embd(self):
        x_batch = tf.reshape(self.x_batch, (-1, 600))
        embeddings = self.model.layers[0](x_batch)
        normalized_embeddings = self.model.layers[1](embeddings)
        reshaped_embeddings = self.model.layers[2](normalized_embeddings)

        saliency = Saliency(self.model, model_modifier=self.model_modifier, clone=True)

        # saliency_map = saliency(self.loss, x_batch, smooth_samples=20, smooth_noise=0.000001)
        saliency_map = saliency(self.loss, reshaped_embeddings)
        heatmap = saliency_map[0]
        avg_token_impact = heatmap.mean(axis=1)

        # reshaping (the make the 1D representation wider)
        avg_token_impact = avg_token_impact.reshape((1, -1)).T
        avg_token_impact = np.array([avg_token_impact for _ in range(10)])
        x_author = tf.reshape(self.x_author, (-1, 1)).numpy()
        x_author = np.array([x_author for _ in range(10)])

        # plotting
        plt.figure(figsize=(18, 6))
        plt.subplot(211)
        plt.title("Source code")
        plt.imshow(x_author)
        plt.subplot(212)
        plt.title("GradCAM map")
        plt.imshow(avg_token_impact)

    def run(self):
        if self.model_name == "conv2d":
            self.run_conv2d()
        elif self.model_name == "embedding":
            self.run_embd()

        plt.savefig("../outputs/vis_{}.png".format(self.model_name))
