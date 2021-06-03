from src.visualization.Visualizer import Visualizer
from keract import get_activations, display_activations
# https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb


class KeractVisualizer(Visualizer):
    def __init__(self, model_name: str, snippet_index: int):
        super().__init__(model_name=model_name, snippet_index=snippet_index)

    def run_conv2d(self):
        self.x_author = self.x_author.numpy().reshape((1, 200, 120))

    def run_embd(self):
        self.x_author = self.x_author.numpy().reshape((1, 600, 1))

    def run(self):
        if self.model_name == "conv2d":
            self.run_conv2d()
        elif self.model_name == "embedding":
            self.run_embd()

        activations = get_activations(self.model, self.x_author, auto_compile=True)

        display_activations(activations, cmap=None, save=True,
                            directory="../outputs/keract_visualization/{}/".format(self.model_name),
                            data_format="channels_last", reshape_1d_layers=False)
