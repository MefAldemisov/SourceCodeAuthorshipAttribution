from models.Classification import Classification
from models.Embedding import Embedding
from models.Linear import Linear
from models.Conv2D import Conv2D

# classification = Classification(make_initial_preprocess=False)
# classification.train(2, 32)

# embedding = Embedding(make_initial_preprocess=False, triplet_type="default")
# print("Embedding model summary", embedding.model.summary(), sep='\n')
# embedding.train(batch_size=128, epochs=1)

# linear = Linear(make_initial_preprocess=False, triplet_type="default")
# linear.train(batch_size=128, epochs=1)

conv2d = Conv2D(make_initial_preprocess=False, triplet_type="default")
conv2d.train(batch_size=128, epochs=1)
