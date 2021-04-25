from models.Classification import Classification
from models.Embedding import Embedding
from models.Linear import Linear
from models.Conv2D import Conv2D
import matplotlib.pyplot as plt


def plot_training(history):
    plt.figure(figsize=(21, 7))
    for i, metric in enumerate(["loss", "accuracy", "recall"]):

        plt.subplot(131 + i)
        plt.title(metric.capitalize())
        plt.plot(history[metric], color='b', label="Training {}".format(metric))
        plt.legend(loc='best', shadow=True)

    plt.savefig("../outputs/training.png")


# classification = Classification(make_initial_preprocess=False)
# history = classification.train(2, 32)

embedding = Embedding(make_initial_preprocess=False, triplet_type="default")
# print("Embedding model summary", embedding.model.summary(), sep='\n')
history = embedding.train(batch_size=128, epochs=1)

# linear = Linear(make_initial_preprocess=False, triplet_type="default")
# history = linear.train(batch_size=128, epochs=1)
plot_training(history)

# conv2d = Conv2D(make_initial_preprocess=False, triplet_type="default")
# conv2d.train(batch_size=128, epochs=1)
