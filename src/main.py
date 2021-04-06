from models.Classification import Classification
from models.Embedding import Embedding
from models.Linear import Linear
from models.Conv2D import Conv2D

# classification = Classification(make_initial_preprocess=False)
# history = classification.train(2, 32)

embedding = Embedding(make_initial_preprocess=False)
print("Embedding model summary", embedding.model.summary(), sep='\n')
embedding.train(batch_size=128, epochs=10)


# linear = Linear(make_initial_preprocess=False)
# linear.train(batch_size=128, epochs=1)

# conv2d = Conv2D(make_initial_preprocess=False)
# conv2d.train(batch_size=128, epochs=1)

