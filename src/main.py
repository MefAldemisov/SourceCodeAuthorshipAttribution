from models.Classification import Classification
from models.Embedding import Embedding
from models.Linear import Linear
from models.Conv2D import Conv2D

# classification = Classification(make_initial_preprocess=False)
# history = classification.train(2, 32)

# embedding = Embedding(make_initial_preprocess=False, triplet_type="default")
# print("Embedding model summary", embedding.model.summary(), sep='\n')
# embedding.train(batch_size=1024, epochs=6)
#
# import tensorflow as tf
#
# a = tf.constant([1, 2, 3, 2])
# a, b = tf.meshgrid(a, a)
# h = tf.math.equal(a, b)
# h1 = tf.boolean_mask(a, h)
# print(a, h, h1)

linear = Linear(make_initial_preprocess=False, triplet_type="default")
linear.train(batch_size=1024, epochs=6)

# conv2d = Conv2D(make_initial_preprocess=False, triplet_type="default")
# conv2d.train(batch_size=128, epochs=1)
