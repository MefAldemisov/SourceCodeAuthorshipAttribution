import abc
import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, name: str,
                 crop: int = 100,
                 make_initial_preprocess: bool = False):
        """
        Interface for all models to be trained

        Parameters:
        -----------
        - name, string - name of the model
        (for debug, model and datasets safe)
        - crop, int - crop size of the width of the vector/ height of the image
        - make_initial_preprocess, bool - if the preprocessing to the json file should be done (requires time)
        """
        self.name = name
        self.crop = crop
        self.make_initial_preprocess = make_initial_preprocess
        self.model = None  # to be substituted with self.create_model()

    @abc.abstractmethod
    def initial_preprocess(self, df_path: str, tmp_dataset_filename: str):
        raise NotImplementedError

    @abc.abstractmethod
    def secondary_preprocess(self, tmp_dataset_filename: str):
        raise NotImplementedError

    @staticmethod
    def _initial_load(df_path: str) -> pd.DataFrame:
        """
        Loading of the dataset and the removal of the additional columns

        :param df_path: str, path to the csv file with initial dataset
        :return: dataset, pandas dataframe
        """
        df = pd.read_csv(df_path)
        # df = df.drop(columns=["round", "task", "solution", "file",
        #                       "full_path", "Unnamed: 0.1", "Unnamed: 0", "lang"])
        df["n_lines"] = df.flines.apply(lambda x: str(x).count("\n"))
        return df

    @staticmethod
    def _user_selection_and_encoding(df: pd.DataFrame,
                                     start_index: int = 0,
                                     length: int = 500) -> pd.DataFrame:
        """
        1. Selects the top[`start_index` : `start_index` + `length`] users, according to the
        number of their files in the dataset
        2. Label encoding of the `username` column is done

        :param df: pd.Dataframe, dataframe to work with
        :param start_index: int, index to start the selection
        :param length: number of authors to be selected
        :return: reduced and modified dataset
        """
        users = df.user.value_counts()[start_index:start_index+length].index
        df = df[df.user.isin(users)]
        le = LabelEncoder()
        df.user = le.fit_transform(df.user)
        return df

    def _crop_to(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 rs1: Tuple[int, ...],
                 rs2: Tuple[int, ...],
                 threshold: int = 80) -> Tuple[np.ndarray, np.ndarray]:

        new_X = []
        new_y = []
        for old_x, old_y in zip(X, y):
            for el in old_x.reshape(rs1):
                if np.count_nonzero(el) > threshold * self.crop // 100:
                    new_X.append(list(el))
                    new_y.append(old_y)

        new_X = np.array(new_X).reshape(rs2)
        new_y = np.array(new_y)
        return new_X, new_y

    def preprocess(self,
                   df_path: str = "../inputs/processed_dfs/cpp_9_tasks_2016.csv",
                   tmp_dataset_dir: str = "../inputs/preprocessed_jsons/") -> Tuple[np.ndarray, np.ndarray,
                                                                                    np.ndarray, np.ndarray]:
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
