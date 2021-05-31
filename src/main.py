from models.Classification import Classification
from models.Embedding import Embedding
from models.Linear import Linear
from models.Conv2D import Conv2D
from visualization.Visualizer import Visualizer
from visualization.KeractVisualizer import KeractVisualizer
# classification = Classification(make_initial_preprocess=False)
# classification.train(2, 32)

# embedding = Embedding(make_initial_preprocess=False)
# print("Embedding model summary", embedding.model.summary(), sep='\n')
# embedding.train(batch_size=128, epochs=1)

# linear = Linear(make_initial_preprocess=False)
# linear.train(batch_size=128, epochs=1)

# conv2d = Conv2D(make_initial_preprocess=False)
# conv2d.train(batch_size=128, epochs=1)

# Visualizer("conv2d").run()
# Visualizer("embedding").run()
KeractVisualizer("conv2d").run()
KeractVisualizer("embedding").run()