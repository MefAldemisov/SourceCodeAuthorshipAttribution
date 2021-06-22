import abc


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

    def create_after_emb(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def create_model(self, **kwargs):
        """
        Creates a model (for triplet loss - the core only)
        The created model should be saved and accessible in self.model
        """
        raise NotImplementedError
