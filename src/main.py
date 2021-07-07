from training.SingleTriplet import SingleTriplet
from training.AvgTriplet import AverageTriplet
from models.Embedding import Embedding
# from models.Conv2D import Conv2D
# from visualization.VisualizerTokenFeatures import VisualizerTokenFeatures
# from visualization.VisualizerCharFeatures import VisualizerCharFeatures

model = Embedding(input_size=100, output_size=50, make_initial_preprocess=False)
SingleTriplet(model=model).train(batch_size=16, epochs=10)
# VisualizerTokenFeatures().run()

# model = Conv2D()
# SingleTriplet(model=model).train(batch_size=16, epochs=1)
# VisualizerCharFeatures().run()
