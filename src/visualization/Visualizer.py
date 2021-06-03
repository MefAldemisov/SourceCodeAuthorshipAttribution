import numpy as np
import pandas as pd
import tensorflow as tf
import sentencepiece as spm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
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

    def loss(self, output: tf.Tensor):
        l = self.hard_triplet_loss(self.y_batch, output)
        return l  # [l for _ in output]

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

    def show_html_conv2d(self, char_impact: np.ndarray):
        color_map = cm.get_cmap("Reds")
        char_impact /= char_impact.max()  # normalization
        with open("../outputs/text_{}.html".format(self.model_name), "a") as file:
            file.write("<div>\n")
            for char_line, impact_line in zip(self.x_author, char_impact):
                for char, impact in zip(char_line, impact_line):
                    if int(char) == 0:
                        continue
                    local_impact = color_map(impact)  # to convert to the [0, 255] range
                    local_impact = [l * 255 for l in local_impact]
                    char = chr(int(char))
                    file.write(
                        "\t<span style='background-color: rgba({}, {}, {}, {})'>{}</span>\n".format(*local_impact,
                                                                                                    char))
                file.write("<br>")
            file.write("<\div>")

    def run_conv2d(self):
        saliency = Gradcam(self.model, model_modifier=lambda m: m, clone=False)

        x_batch = tf.reshape(self.x_batch, (-1, 200, 120, 1))
        # saliency_map = saliency(self.loss, x_batch, smooth_samples=20, smooth_noise=0.2)
        saliency_map = saliency(self.loss, x_batch, penultimate_layer=-1)
        heatmap = saliency_map[self.snippet_index]
        # plotting
        plt.figure(figsize=(18, 6))
        plt.subplot(121)
        plt.title("Source code")
        plt.imshow(self.x_author)
        plt.subplot(122)
        plt.title("GradCAM map")
        plt.imshow(heatmap)
        self.show_html_conv2d(heatmap)

    def show_html_embd(self, token_impact: np.ndarray):
        sp = spm.SentencePieceProcessor(model_file='../inputs/embd/sentencepiece_bpe.model')
        color_map = cm.get_cmap("Reds")
        token_impact /= token_impact.max()  # normalization
        with open("../outputs/text_{}.html".format(self.model_name), "a") as file:
            file.write("<div>\n")
            for token, impact in zip(self.x_author, token_impact):
                local_impact = color_map(impact)[0] * 255  # to convert to the [0, 255] range
                word = sp.decode(int(token))
                file.write("\t<span style='background-color: rgba({}, {}, {}, {})'>{}</span>\n".format(*local_impact,
                                                                                                       word))
            file.write("</div>")

    def run_embd(self):
        x_batch = tf.reshape(self.x_batch, (-1, 600))
        embeddings = self.model.layers[0](x_batch)
        normalized_embeddings = self.model.layers[1](embeddings)
        reshaped_embeddings = self.model.layers[2](normalized_embeddings)

        saliency = Saliency(self.model, model_modifier=self.model_modifier, clone=True)

        saliency_map = saliency(self.loss, reshaped_embeddings)
        heatmap = saliency_map[self.snippet_index]
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
        self.show_html_embd(avg_token_impact[0])

    def run(self):
        if self.model_name == "conv2d":
            self.run_conv2d()
        elif self.model_name == "embedding":
            self.run_embd()

        plt.savefig("../outputs/vis_{}.png".format(self.model_name))
