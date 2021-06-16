from src.visualization.base.Visualizer import Visualizer
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from src.models.data_processing.CharFeatures import CharFeatures

class VisualizerCharFeatures(Visualizer):
    def __init__(self):
        data_loader = CharFeatures(name="conv2d", make_initial_preprocess=False)
        super().__init__(model_name="conv2d",
                         data_loader=data_loader,
                         snippet_index = 0)

    def read_dataset(self):
        # load dataset
        df = pd.read_json("../inputs/preprocessed_jsons/{}_train.json".format(self.model_name))
        y = np.array(df.username)
        # read X
        file = open("../inputs/preprocessed_jsons/{}_train.txt".format(self.model_name), "rb")
        x = np.load(file)
        file.close()
        return x, y

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
                        "<span style='background-color: rgba({}, {}, {}, {})'>{} </span>".format(*local_impact, char))
                file.write("<br>")
            file.write("</pre></div>")

    def run(self):
        heatmap = self.get_heatmap((-1, 200, 120), (200, 120, 1))[self.snippet_index]
        self.plot_image(self.x_author, heatmap, [121, 122])
        self.show_html(heatmap, self.x_author)
