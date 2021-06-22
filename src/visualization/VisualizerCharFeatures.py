import numpy as np
import tensorflow as tf
import matplotlib.cm as cm

from src.models.Conv2D import Conv2D
from src.visualization.base.Visualizer import Visualizer
from src.models.data_processing.CharFeatures import CharFeatures


class VisualizerCharFeatures(Visualizer):
    def __init__(self):
        data_loader = CharFeatures(name="conv2d", make_initial_preprocess=False, crop=20)
        super().__init__(model_name="conv2d",
                         data_loader=data_loader,
                         snippet_index=0)

    def show_html(self, char_impact: np.ndarray, initial_tokens: np.ndarray, label_index: int = 0):
        color_map = cm.get_cmap("Reds")
        char_impact /= np.max(char_impact)  # normalization
        with open("../outputs/text_{}.html".format(self.model_name), "w") as file:
            file.write("<div><pre>\n")
            file.write("<h2>Author #{}</h2>\n".format(label_index))
            for char_line, impact_line in zip(initial_tokens, char_impact):
                for char, impact in zip(char_line, impact_line):
                    if int(char) == 0:
                        continue
                    local_impact = self.get_color(color_map, impact)
                    char = chr(int(char))
                    file.write(
                        "<span style='background-color: rgba({}, {}, {}, {})'>{}</span>".format(*local_impact, char))
                file.write("<br>")
            file.write("</pre></div>")

    def run(self):
        input_layer = tf.keras.layers.Input((20, 120, 100, 1))
        output = Conv2D(crop=20).create_after_emb(input_layer)
        clean_model = tf.keras.models.Model(input_layer, output)

        heatmap = self.get_heatmap((-1, 20 * 120, 1), clean_model=clean_model, layers_to_cut=3)[self.snippet_index]
        heatmap = heatmap.mean(axis=2)  # averaging of the embedding's output
        self.x_author = self.x_author.numpy().reshape(20, 120)
        self.plot_image(self.x_author, heatmap, [121, 122])
        self.show_html(heatmap, self.x_author)
