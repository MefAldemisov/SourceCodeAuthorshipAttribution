from models.Embedding import Embedding
from models.Conv2D import Conv2D
from visualization.Visualizer import Visualizer
from visualization.KeractVisualizer import KeractVisualizer

# embedding = Embedding(make_initial_preprocess=False)
# embedding.train(batch_size=128, epochs=1)

# conv2d = Conv2D(make_initial_preprocess=False)
# conv2d.train(batch_size=128, epochs=1)

Visualizer("conv2d").run()
Visualizer("embedding").run()

# KeractVisualizer("conv2d").run()
# KeractVisualizer("embedding").run()
