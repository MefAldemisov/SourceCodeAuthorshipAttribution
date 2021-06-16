import numpy as np
import pandas as pd
import sentencepiece as spm

from sklearn.model_selection import train_test_split
from src.models.data_processing.base.DataLoading import DataLoader


class TokenFeatures(DataLoader):

    def __init__(self,
                 name: str,
                 input_size: int = 100,
                 crop=None,
                 make_initial_preprocess: bool = True):
        self.input_size = input_size
        crop = input_size if crop is None else crop
        super().__init__(crop=crop, name=name, make_initial_preprocess=make_initial_preprocess)

    def initial_preprocess(self, df_path, tmp_dataset_filename):
        df = self._initial_load(df_path)
        df = df[(df.n_lines > 0)]
        # tokenize
        sp = spm.SentencePieceProcessor(model_file='../inputs/embd/sentencepiece_bpe.model')
        df.index = np.arange(len(df))
        df["n_tokens"] = df.flines.apply(lambda x: len(sp.encode(x)))
        df = df[df.n_tokens <= self.input_size]
        # reduce size
        df = self._user_selection_and_encoding(df, 50, 500)
        # long saving
        df["tokens"] = df.flines.apply(lambda x: sp.encode(x))
        dataset = df[["username", "tokens"]]
        # shuffle dataset
        dataset = dataset.sample(frac=1)

        def rsh(x):
            arr = np.array(x)
            arr.resize((self.input_size, 1))  # probably, error
            return list(arr)

        dataset.tokens = dataset.tokens.apply(rsh)
        dataset.to_json(tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename: str):
        dataset = pd.read_json(tmp_dataset_filename)

        X = dataset.tokens.values
        X = np.array(list(X)).reshape((-1, self.input_size))

        y = np.array(dataset.username)
        X, y = self._crop_to(X, y, rs1=(-1, self.crop), rs2=(-1, self.crop, 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
