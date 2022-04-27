import torch
import tqdm

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaModel


def generate_data(df_path: str, data_path: str, INPUT_SIZE: int, BATCH_SIZE: int):
    df = pd.read_csv(df_path)
    # df = df.drop(columns=["round", "task", "solution", "file",
    #                       "full_path", "Unnamed: 0.1", "Unnamed: 0", "lang"])
    # df["n_lines"] = df.flines.apply(lambda x: str(x).count("\n"))
    # df = df[(df.n_lines > 0)]

    # def _insert_tokens(x: str):
    #     x = x.replace("\n", " NLN ")
    #     x = x.replace("\t", " TAB ")
    #     x = x.replace(" ", " SPC ")
    #     return x
    #
    # df.flines = df.flines.apply(_insert_tokens)

    # load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    df.index = np.arange(len(df))
    le = LabelEncoder()
    df.user = le.fit_transform(df.user)
    df['tokens'] = df.flines.apply(lambda x: tokenizer
                                   .convert_tokens_to_ids(tokenizer.tokenize(x)))

    dataset = df[["user", "tokens", "task"]]
    # shuffle dataset
    dataset = dataset.sample(frac=1)

    X = dataset.tokens.values

    def fill_zeros(arr):
        arr = np.array(arr)
        if INPUT_SIZE > arr.shape[0]:
            arr = np.pad(arr, (0, INPUT_SIZE - arr.shape[0]), 'constant')
        else:
            arr = arr[:INPUT_SIZE]
        return arr.reshape(INPUT_SIZE, 1).tolist()

    X = np.array([fill_zeros(x) for x in X])
    X = X.reshape((-1, INPUT_SIZE))
    y = np.array(dataset.user)
    tasks = np.array(dataset.task)
    train_indexes = np.where(tasks < 7)[0]
    test_indexes = np.where(tasks >= 7)[0]
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    embedding_model = RobertaModel.from_pretrained("microsoft/codebert-base")

    def get_embedding(data):
        emb = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, data.shape[0], BATCH_SIZE)):
                batch = data[i: i+BATCH_SIZE]
                new_part = embedding_model(torch.from_numpy(batch)).last_hidden_state
                emb = [*emb, *new_part]
        return emb

    x_emb = get_embedding(X_test)
    np.save(data_path + 'x_train.np', X_train)
    np.save(data_path + 'y_test.np', y_test)
    np.save(data_path + 'y_train.np', y_train)
    np.save(data_path + 'x_test.np', X_test)
    torch.save(torch.cat(x_emb), data_path + 'test_tensor.pt')
    x_train_emb = get_embedding(X_train)
    torch.save(torch.cat(x_train_emb), data_path + 'train_tensor.pt')
