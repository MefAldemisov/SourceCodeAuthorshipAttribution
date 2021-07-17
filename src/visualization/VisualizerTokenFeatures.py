from visualization.base.Visualizer import Visualizer
import numpy as np
# import pandas as pd
# import sentencepiece as spm
import tensorflow_text as text
import matplotlib.cm as cm
import tensorflow as tf

from models.data_processing.TokenFeatures import TokenFeatures
from models.Embedding import Embedding


class VisualizerTokenFeatures(Visualizer):
    def __init__(self):
        data_loader = TokenFeatures(name="embedding", make_initial_preprocess=False,
                                    input_size=800, crop=200)
        super().__init__(model_name="embedding",
                         data_loader=data_loader,
                         snippet_index=0)

    def show_html(self, token_impact: np.ndarray, initial_tokens: np.ndarray, label_index: int = 0):
        # sp = spm.SentencePieceProcessor(model_file='../inputs/embd/sentencepiece_bpe.model')
        tokenizer = text.BertTokenizer("../inputs/bert_tokens.model")

        arr = np.zeros(99757)
        color_map = cm.get_cmap("Reds")

        token_impact /= np.max(token_impact)  # normalization to range [0; 1]
        with open("../outputs/text_{}.html".format(self.model_name), "a") as file:
            file.write("<div><h2>Author #{}</h2>\n".format(label_index))
            for token, impact in zip(initial_tokens, token_impact):
                # if impact > 0.5:
                arr[int(token)] += 1
                local_impact = self.get_color(color_map, impact)
                word = tokenizer.detokenize([[int(token)]]).to_list()[0][0].decode("utf-8")
                # special tokens
                if word == "TAB":
                    word = "&emsp"
                elif word == "SPC":
                    word = "&nbsp"
                elif word == "NLN":
                    file.write("<br>")
                    continue

                file.write("<span style='background-color: rgba({}, {}, {}, {})'>{}</span>"
                           .format(*local_impact, word))

            file.write("</div>")
        return arr

    def run_embd(self, heatmap: np.ndarray, x_author: tf.Tensor, y_author: int):
        avg_token_impact = heatmap.mean(axis=1)
        frq = self.show_html(avg_token_impact, x_author.numpy(), y_author)

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
        # create a new file
        with open("../outputs/text_{}.html".format(self.model_name), "w") as file:
            file.write("<main style='word-wrap: break-word;'>")

        snippet_length = 200  # initially 600

        input_layer = tf.keras.layers.Input((snippet_length, 100, 1))
        output = Embedding().create_after_emb(input_layer)
        clean_model = tf.keras.models.Model(input_layer, output)

        heatmap = self.get_heatmap(target_shape=(-1, snippet_length), clean_model=clean_model, layers_to_cut=3)
        frequency = np.zeros((self.x_batch.shape[0], 99757))
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
