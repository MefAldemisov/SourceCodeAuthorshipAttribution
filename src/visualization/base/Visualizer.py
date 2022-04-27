import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.data_processing.base.DataLoading import DataLoader
from training.AvgTriplet import AverageTriplet
from matplotlib.colors import Colormap
from tensorflow.keras.models import Model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from typing import Tuple, List
from tf_keras_vis.scorecam import ScoreCAM

# https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb
# https://docs.seldon.io/projects/alibi/en/latest/examples/integrated_gradients_imdb.html


class Visualizer:
    def __init__(self,
                 model_name: str,
                 data_loader: DataLoader,
                 snippet_index: int = 0):
        self.model_name = model_name
        self.snippet_index = snippet_index

        self.model = tf.keras.models.load_model('../outputs/{}_49.h'.format(model_name))
        all_x, _, all_y, _ = data_loader.secondary_preprocess("../inputs/preprocessed_jsons/{}_train.json"
                                                              .format(model_name))
        self.triplet_type = AverageTriplet(self.model)
        self.batch_size = 100
        self.x_batch, self.y_batch = self.triplet_type.batch_generator(all_x, all_y, self.batch_size)
        self.x_author = self.x_batch[self.snippet_index]

    def loss(self, output: tf.Tensor) -> tf.Tensor:
        loss_value = self.triplet_type.triplet_loss([self.y_batch, output])
        return loss_value  # [loss_value for _ in output]

    @staticmethod
    def get_color(color_map: Colormap, impact: np.ndarray):
        # to convert to the [0, 255] range
        impact_color = color_map(impact)
        color_255 = [color * 255 for color in impact_color]
        return color_255

    def show_html(self,
                  char_impact: np.ndarray,
                  initial_tokens: np.ndarray,
                  label_index: int = 0):
        raise NotImplementedError

    def get_heatmap(self,
                    target_shape: Tuple,
                    clean_model: Model,
                    layers_to_cut: int = 0) -> np.ndarray:
        """
        Generates a heatmap for a given batch of source files using `self.x_batch` as an input

        :param target_shape: should be the same as the shape to the input layer of the model
        :param clean_model: empty model (same architecture as the model in the constructor) to work with
        :param layers_to_cut: the number of layers, which depend on the embedding layer
        :return: a heatmap (Saliency map) for a given batch of source files using `self.x_batch` as an input
        """

        def model_modifier(m: Model) -> Model:
            model = clean_model
            for layer_index in range(layers_to_cut, len(m.layers)):
                weights = m.layers[layer_index].get_weights()
                if weights:  # else - dropout or normalization
                    model.layers[layer_index-layers_to_cut+1].set_weights(weights)
            return model

        x_batch = tf.reshape(self.x_batch, target_shape)
        layers = self.model.layers
        for layer in layers[:layers_to_cut]:
            x_batch = layer(x_batch)

        modified_model = model_modifier(self.model)
        saliency = Saliency(modified_model, model_modifier=None, clone=False)
        saliency_map = saliency(self.loss, x_batch)
        heatmap = saliency_map
        assert heatmap.max() != 0.0, "the loss value is too small"
        return heatmap

    def plot_image(self, x_author: np.ndarray, impact_map: np.ndarray, axis: List[int]):
        plt.figure(figsize=(18, 6))
        plt.subplot(axis[0])
        plt.title("Source code")
        plt.imshow(x_author)
        plt.subplot(axis[1])
        plt.title("Saliency map")
        plt.imshow(impact_map)
        plt.savefig("../outputs/vis_{}.png".format(self.model_name))
        plt.close()

    def run(self):
        raise NotImplementedError
