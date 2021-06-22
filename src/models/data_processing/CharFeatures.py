import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from src.models.data_processing.base.DataLoading import DataLoader


class CharFeatures(DataLoader):
    def __init__(self,
                 name: str,
                 img_x: int = 120,
                 img_y: int = 200,
                 crop=None,
                 make_initial_preprocess: bool = True):
        self.img_x, self.img_y = img_x, img_y
        crop = self.img_y if crop is None else crop
        super().__init__(crop=crop, name=name, make_initial_preprocess=make_initial_preprocess)

    @staticmethod
    def _path_to_x(y_path: str):
        x_path = y_path[:-4]  # "json" removed
        x_path += "txt"
        return x_path

    def initial_preprocess(self, df_path: str, tmp_dataset_filename: str):
        df = self._initial_load(df_path)
        df = df[(df.n_lines >= 30) & (df.n_lines < self.img_y)]  # there should be enough loc for 2D convolution

        def max_cpl(file: str):
            """"
            Max chars per line
            """
            lines = file.split('\n')
            max_ch = 0
            for line in lines:
                if len(line) > max_ch:
                    max_ch = len(line)
            return max_ch

        df["max_cpl"] = df.flines.apply(max_cpl)
        df = df[df.max_cpl <= self.img_x]

        df = self._user_selection_and_encoding(df, 0, 500)

        def to_vector(file: str):
            lines = file.split("\n")
            res = np.zeros((self.img_y, self.img_x), dtype=int)
            for i in range(len(lines)):
                if i >= self.img_y:
                    break
                line = lines[i]
                for j in range(len(line)):
                    res[i][j] = ord(line[j])
            return res.tolist()

        X = df.flines.apply(to_vector).values
        X = np.array([np.array(x) for x in X])
        X = X.reshape((-1, self.img_y, self.img_x))
        # save X and y separately
        x_file = open(self._path_to_x(tmp_dataset_filename), "wb")
        np.save(x_file, X)
        x_file.close()
        dataset = df[["username"]]
        dataset.to_json(tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename: str):
        df = pd.read_json(tmp_dataset_filename)
        y = np.array(df.username)
        # read X
        file = open(self._path_to_x(tmp_dataset_filename), "rb")
        X = np.load(file)
        file.close()
        # chunking
        X, y = self._crop_to(X, y,
                             rs1=(-1, self.crop, self.img_x),
                             rs2=(-1, self.crop * self.img_x, 1))

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
