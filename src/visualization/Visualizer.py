import matplotlib.colors
import numpy as np
import pandas as pd
import tensorflow as tf
import sentencepiece as spm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import Colormap
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from typing import Tuple, List
from src.models.Triplet import Triplet
from tf_keras_vis.scorecam import ScoreCAM

# https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb
# https://docs.seldon.io/projects/alibi/en/latest/examples/integrated_gradients_imdb.html


class Visualizer(Triplet):
    def __init__(self, model_name: str, snippet_index: int = 0):
        super().__init__(name="none", make_initial_preprocess=False)
        self.model_name = model_name
        self.snippet_index = snippet_index
        self.model = tf.keras.models.load_model('../outputs/{}.h'.format(model_name))
        all_x, all_y = self.read_conv2d_dataset() if model_name == "conv2d" else self.read_embedding_dataset()
        self.x_batch, self.y_batch = self._batches_generator(all_x, all_y, 128)
        self.x_author = self.x_batch[self.snippet_index]

    @staticmethod
    def create_model():
        return None

    def loss(self, output: tf.Tensor) -> float:
        loss_value = self.hard_triplet_loss(self.y_batch, output)
        return loss_value  # [loss_value for _ in output]

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
        # dataset = pd.read_json("../inputs/preprocessed_jsons/{}_train.json".format(self.model_name))
        dataset = pd.read_json("../inputs/preprocessed_jsons/overfit.json".format(self.model_name))
        X = dataset.tokens.values
        X = np.array(list(X)).reshape((-1, 600))

        y = np.array(dataset.username)
        return X, y

    @staticmethod
    def get_color(color_map: Colormap, impact: np.ndarray):
        # to convert to the [0, 255] range

        impact_color = color_map(impact)
        color_255 = [color * 255 for color in impact_color]
        return color_255

    def show_html_conv2d(self, char_impact: np.ndarray, initial_tokens: np.ndarray):
        color_map = cm.get_cmap("Reds")
        char_impact /= np.max(char_impact)  # normalization
        with open("../outputs/text_{}.html".format(self.model_name), "w") as file:
            file.write("<div><pre>\n")
            for char_line, impact_line in zip(initial_tokens, char_impact):
                for char, impact in zip(char_line, impact_line):
                    if int(char) == 0:
                        continue
                    local_impact = self.get_color(color_map, impact)
                    char = chr(int(char))
                    file.write(
                        "<span style='background-color: rgba({}, {}, {}, {})'>{} </span>".format(*local_impact, char))
                file.write("<br>")
            file.write("</pre></div>")

    def show_html_embd(self, token_impact: np.ndarray, initial_tokens: np.ndarray, label_index: int = 0):
        sp = spm.SentencePieceProcessor(model_file='../inputs/embd/sentencepiece_bpe.model')
        arr = np.zeros(99757)
        color_map = cm.get_cmap("Reds")
        token_impact /= np.max(token_impact)  # normalization to range [0; 1]
        with open("../outputs/text_{}.html".format(self.model_name), "a") as file:
            file.write("<div><h2>Author #{}</h2>\n".format(label_index))
            for token, impact in zip(initial_tokens, token_impact):
                # if impact > 0.5:
                arr[int(token)] += 1
                local_impact = self.get_color(color_map, impact)
                word = sp.decode(int(token))

                file.write("<span style='background-color: rgba({}, {}, {}, {})'>{}</span>".format(*local_impact,
                                                                                                       word))
            file.write("</div>")
        return arr

    def get_heatmap(self,
                    target_shape: Tuple,
                    cut_shape: Tuple = (600, 1),
                    layers_to_cut: int = 0) -> np.ndarray:
        """
        Generates a heatmap for a given batch of source files using `self.x_batch` as an input

        :param target_shape: should be the same as the shape to the input layer of the model
        :param cut_shape: models, which utilize the `tf.keras.layers.Embedding` layer should be split into
        two parts, where the first part is not differentiable and the second one is. The `cut_shape` parameter
        specifies the size of the input to the second part, which is the same as the size of output to the
        first one.
        :param layers_to_cut: the number of layers, which depend on the embedding layer
        :return: a heatmap (Saliency map) for a given batch of source files using `self.x_batch` as an input
        """

        def model_modifier(m: Model) -> Model:
            new_model = Sequential()
            new_model.add(Input(cut_shape))
            for new_layer in m.layers[layers_to_cut:]:
                new_model.add(new_layer)
            return new_model

        x_batch = tf.reshape(self.x_batch, target_shape)
        layers = self.model.layers
        for layer in layers[:layers_to_cut]:
            x_batch = layer(x_batch)

        saliency = Saliency(self.model, model_modifier=model_modifier, clone=False)
        saliency_map = saliency(self.loss, x_batch)
        heatmap = saliency_map
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

    def run_embd(self, heatmap: np.ndarray, x_author: tf.Tensor, y_author: int):
        avg_token_impact = heatmap.mean(axis=1)
        frq = self.show_html_embd(avg_token_impact, x_author.numpy(), y_author)

        # reshaping (the make the 1D representation wider)

        def make_wider(arr: np.ndarray, width: int = 10) -> np.ndarray:
            arr = arr.reshape((-1, 1))
            return np.array([arr for _ in range(width)])

        avg_token_impact = make_wider(avg_token_impact)
        x_author = make_wider(x_author.numpy())
        # plotting
        self.plot_image(x_author, avg_token_impact, [211, 212])
        return frq

    def run(self):
        if self.model_name == "conv2d":
            heatmap = self.get_heatmap((-1, 200, 120), (200, 120, 1), 5)[self.snippet_index]
            self.plot_image(self.x_author, heatmap, [121, 122])
            self.show_html_conv2d(heatmap, self.x_author)
        elif self.model_name == "embedding":
            # create a new file
            with open("../outputs/text_{}.html".format(self.model_name), "w") as file:
                file.write("<main>")

            heatmap = self.get_heatmap((-1, 600), (600, 100, 1), 3)
            frequency = np.zeros((self.x_batch.shape[0],  99757))
            for i, (h, x, y) in enumerate(zip(heatmap, self.x_batch, self.y_batch)):
                frequency[i] = self.run_embd(h, x, y)

            # close file
            with open("../outputs/text_{}.html".format(self.model_name), "a") as file:
                file.write("</main>")

            # create dataframe and rename columns to token values
            # sp = spm.SentencePieceProcessor(model_file='../inputs/embd/sentencepiece_bpe.model')
            # df = pd.DataFrame(frequency, index=self.y_batch.numpy().T.tolist()[0],
            #                   columns=[sp.decode(int(token)) for token in range(0, frequency.shape[1])])


            # print("The most common token within a batch:", sp.decode(int(frequency.max(axis=0).argmax())),
            #       "its frequency:", frequency.max(axis=0).max())
            #
            # token_values = frequency.max(axis=1)
            # token_indexes = frequency.argmax(axis=1)
            # for i, y in enumerate(df.index):
            #     print("user", y, "the most frequent token",
            #           sp.decode(int(token_indexes[i])),
            #           "(", token_values[i], ")")

            # df.to_csv("../outputs/embedding-frequency.csv")
