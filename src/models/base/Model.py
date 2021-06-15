import abc

import tqdm
import numpy as np
import tensorflow as tf

from src.training.TrainingCallback import TestCallback
from tensorflow.keras import optimizers, callbacks
# according to the documentation, BallTree is more efficient in high-dimensional case
from sklearn.neighbors import BallTree

class Model:
    def __init__(self):
        """
        Interface for all models to be trained

        Parameters:
        -----------
        - name, string - name of the model
        (for debug, model and datasets safe)
        """
        self.model = None  # to be substituted with self.create_model()

    # @abc.abstractmethod
    # def create_embedding_part_of_the_model(self, **kwargs):
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def create_second_part_of_the_model(self, **kwargs):
    #     raise NotImplementedError

    @abc.abstractmethod
    def create_model(self, **kwargs):
        """
        Creates a model (for triplet loss - the core only)
        The created model should be saved and accessible in self.model
        """
        raise NotImplementedError
