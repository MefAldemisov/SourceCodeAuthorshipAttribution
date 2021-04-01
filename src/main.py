from models.Classification import Classification
from models.Embedding import Embedding
from visualization.visualization import plot_training
# restore the training of the linear model
classification = Classification(make_initial_preprocess=False)
history = classification.train(2, 32)

embedding = Embedding(make_initial_preprocess=False)
print("Embedding model summary", embedding.model.summary(), sep='\n')
embedding.train(batch_size=128, epochs=1)

