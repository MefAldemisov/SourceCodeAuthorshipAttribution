import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_text as text
from typing import List
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
# from sklearn.model_selection import train_test_split
from models.data_processing.base.DataLoading import DataLoader


bert_vocab_args = {
    "vocab_size": 80000,
    "reserved_tokens": ["TAB", "SPC", "NLN"],
    "bert_tokenizer_params": {},
    "learn_params": {},
}


class TokenFeatures(DataLoader):

    def __init__(self,
                 name: str,
                 input_size: int = 600,
                 crop=200,
                 make_initial_preprocess: bool = True):
        self.input_size = input_size
        crop = input_size if crop is None else crop
        super().__init__(crop=crop, name=name, make_initial_preprocess=make_initial_preprocess)

    @staticmethod
    def _write_vocab_file(filepath: str, vocab: List[str]):
        with open(filepath, "w") as f:
            for token in vocab:
                print(token, file=f)

    @staticmethod
    def _insert_tokens(x: str):
        x = x.replace("\n", " NLN ")
        x = x.replace("\t", " TAB ")
        x = x.replace(" ", " SPC ")
        return x

    def initial_preprocess(self, df_path: str, tmp_dataset_filename: str):
        df = self._initial_load(df_path)
        df = df[(df.n_lines > 0)]
        # tokenize. requires time (approx 1h)

        df.flines = df.flines.apply(self._insert_tokens)
        print("updated")
        text_dataset = tf.data.Dataset.from_tensor_slices(df.flines.values)

        vocab = bert_vocab.bert_vocab_from_dataset(
            text_dataset,
            **bert_vocab_args
        )
        self._write_vocab_file("../inputs/bert_tokens.model", vocab)
        print("saved")
        # read the tokenizer
        tokenizer = text.BertTokenizer("../inputs/bert_tokens.model")

        # reduce the size of the dataset according to the n_tokens
        df.index = np.arange(len(df))
        df["n_tokens"] = df.flines.apply(lambda x: tokenizer.tokenize(x).shape[0])
        df = df[df.n_tokens <= self.input_size]
        # reindex
        df.index = np.arange(len(df))
        # reduce size
        df = self._user_selection_and_encoding(df, 50, 450)
        # long saving
        # The issue is that `tokenizer.tokenize()` do not always return a shape (-1, 1).
        # Some elements of the result of the function could be a list, e.g. [[2929, 8524]].
        # >> tokenizer.detokenize([[2929, 8524]])
        # < tf.RaggedTensor[[b'visdist']] >
        # >> tokenizer.detokenize([[2929]])
        # < tf.RaggedTensor[[b'vis']] >
        # >> tokenizer.detokenize([[8524]])
        # < tf.RaggedTensor[[b'##dist']] >
        # I have decided to flatten these lists.
        df["tokens"] = df.flines.apply(lambda x: tokenizer.tokenize(x).to_list())

        df.tokens = df.tokens.apply(lambda x: list(pd.core.common.flatten(x)))
        dataset = df[["user", "tokens", "task"]]
        # shuffle dataset
        dataset = dataset.sample(frac=1)

        def rsh(x):
            arr = np.array(x)
            arr = np.resize(arr, (self.input_size, 1))
            return arr.tolist()

        dataset.tokens = dataset.tokens.apply(rsh)
        dataset.to_json(tmp_dataset_filename)

    def secondary_preprocess(self, tmp_dataset_filename: str):
        dataset = pd.read_json(tmp_dataset_filename)

        X = dataset.tokens.values
        X = np.array([np.array(x).reshape(self.input_size, 1).tolist() for x in X])
        X = X.reshape((-1, self.input_size))

        y = np.array(dataset.user)
        tasks = np.array(dataset.task)
        train_indexes = np.where(tasks < 7)[0]
        test_indexes = np.where(tasks >= 7)[0]
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

        X_train, y_train = self._crop_to(X_train, y_train, rs1=(-1, self.crop), rs2=(-1, self.crop, 1))
        X_test, y_test = self._crop_to(X_test, y_test, rs1=(-1, self.crop), rs2=(-1, self.crop, 1))
        self.input_size = self.crop

        return X_train, X_test, y_train, y_test
