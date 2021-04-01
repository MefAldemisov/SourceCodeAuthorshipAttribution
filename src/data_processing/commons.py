import numpy as np
import pandas as pd
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder

def std_initial_preprocess(input_length, df_path, tmp_dataset_filename):
    df = pd.read_csv(df_path)
    df = df.drop(columns=["round", "task", "solution", "file", "full_path", "Unnamed: 0.1", "Unnamed: 0", "lang"])
    df["n_lines"] = df.flines.apply(lambda x: str(x).count("\n"))
    df = df[(df.n_lines > 0)]
    # tokenize
    sp = spm.SentencePieceProcessor(model_file='../inputs/embd/sentencepiece_bpe.model')
    df.index = np.arange(len(df))
    df["n_tokens"] = df.flines.apply(lambda x: len(sp.encode(x)))
    df = df[df.n_tokens <= input_length]
    users = df.username.value_counts()[50:550].index
    df = df[df.username.isin(users)]
    df["tokens"] = df.flines.apply(lambda x: sp.encode(x))
    dataset = df[["username", "tokens"]]
    le = LabelEncoder()
    dataset.username = le.fit_transform(dataset.username)
    # shuffle dataset
    dataset = dataset.sample(frac=1)

    def rsh(x):
        arr = np.array(x)
        arr.resize(input_length)
        return list(arr)

    dataset.tokens = dataset.tokens.apply(rsh)
    dataset.to_json(tmp_dataset_filename)