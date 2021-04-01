import abc


class Model:
    def __init__(self, name: str, make_initial_preprocess: bool = True):
        """
        Interface for all models to be trained

        Parameters:
        -----------
        - name, string - name of the model
        (for debug, model and datasets safe)
        """
        self.name = name
        self.make_initial_preprocess = make_initial_preprocess
        self.model = None  # to be substituted with self.create_model()

    @abc.abstractmethod
    def initial_preprocess(self, df_path, tmp_dataset_filename):
        raise NotImplementedError

    @abc.abstractmethod
    def secondary_preprocess(self, tmp_dataset_filename):
        raise NotImplementedError

    def preprocess(self, df_path: str = "../inputs/processed_dfs/py_df.csv",
                   tmp_dataset_dir: str = "../inputs/preprocessed_jsons/"):
        """
        Preprocessing of the give dataset for the given model.
        It should include some model-specific things, as reshaping,
        tokenization, cuts e.t.c

        Parameters:
        -----------
        df_path, str - path for .csv file, data which will be preprocessed by the
        `initial_preprocess` method

        tmp_dataset_dir, str - path for .json file, result of the
        `initial preprocessing` (maximal preprocessing in the DataFrame format)

        Result:
        -------
        X_train, X_test, y_train, y_test - values to be used by the train method
        """
        tmp_dataset_filename = tmp_dataset_dir + self.name + "_train.json"
        if self.make_initial_preprocess:
            self.initial_preprocess(df_path, tmp_dataset_filename)
        return self.secondary_preprocess(tmp_dataset_filename)

    @abc.abstractmethod
    def create_model(self):
        """
        Creates a model (for triplet loss - the core only)
        The created model should be saved and accessible in self.model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        """
        Training of the model
        """
        raise NotImplementedError
