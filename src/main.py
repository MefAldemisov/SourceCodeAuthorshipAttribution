from training.SingleTriplet import SingleTriplet
# from training.AvgTriplet import AverageTriplet
from models.Embedding import Embedding
# from models.Conv2D import Conv2D
# from visualization.VisualizerTokenFeatures import VisualizerTokenFeatures
# from visualization.VisualizerCharFeatures import VisualizerCharFeatures
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

model = Embedding(input_size=800, crop=800, output_size=50, make_initial_preprocess=True)
SingleTriplet(model=model).train(batch_size=16, epochs=50, epoch_start=0, step_start=0)
# VisualizerTokenFeatures().run()

# model = Conv2D()
# SingleTriplet(model=model).train(batch_size=16, epochs=1)
# VisualizerCharFeatures().run()
