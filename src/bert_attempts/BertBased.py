from transformers import RobertaTokenizer, RobertaModel
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from AccuracyEvaluator import AccuracyEvaluator
from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder

# with reference to https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

# -------------------------- constants
df_path = '../../inputs/processed_dfs/cpp_9_tasks_2016.csv'
tmp_dataset_dir = "../../inputs/preprocessed_jsons/"
tmp_dataset_filename = tmp_dataset_dir + 'bert' + "_train.json"

INPUT_SIZE = 512 # 514 tokens, maximum for bert
OUTPUT_SIZE = 256
N_EPOCHS = 100
BATCH_SIZE = 16

# -------------------------- load data
df = pd.read_csv(df_path)
# df = df.drop(columns=["round", "task", "solution", "file",
#                       "full_path", "Unnamed: 0.1", "Unnamed: 0", "lang"])
# df["n_lines"] = df.flines.apply(lambda x: str(x).count("\n"))
# df = df[(df.n_lines > 0)]


def _insert_tokens(x: str):
    x = x.replace("\n", " NLN ")
    x = x.replace("\t", " TAB ")
    x = x.replace(" ", " SPC ")
    return x

df.flines = df.flines.apply(_insert_tokens)

# load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
df.index = np.arange(len(df))
le = LabelEncoder()
df.user = le.fit_transform(df.user)
df['tokens'] = df.flines.apply(lambda x:
                              tokenizer.convert_tokens_to_ids(
                                  tokenizer.tokenize(x)))

dataset = df[["user", "tokens", "task"]]
# shuffle dataset
dataset = dataset.sample(frac=1)

X = dataset.tokens.values

def fillZeros(arr):
    arr = np.array(arr)
    if INPUT_SIZE > arr.shape[0]:
        arr = np.pad(arr, (0, INPUT_SIZE - arr.shape[0]), 'constant')
    else:
        arr = arr[:INPUT_SIZE]
    return arr.reshape(INPUT_SIZE, 1).tolist()

X = np.array([fillZeros(x) for x in X])
X = X.reshape((-1, INPUT_SIZE))
y = np.array(dataset.user)
tasks = np.array(dataset.task)
train_indexes = np.where(tasks < 7)[0]
test_indexes = np.where(tasks >= 7)[0]
X_train, X_test = X[train_indexes], X[test_indexes]
y_train, y_test = y[train_indexes], y[test_indexes] # 244 unique person

# -------------------------- model architecture

# let's do just a simple thing

# 1. embedding from bert -> INPUT_SIZE * 768
# 2. convolution (5*768)
# 3. fully-connected 500
# 4. fully connected 100

# 1. pretrained part

# train loader
class GCJ:
    def __init__(self, X_train, y_train, batch_size = BATCH_SIZE):
        self.x = X_train
        self.y = y_train
        self.batch_size = batch_size

    def batch_generator(self, model, tree):
        n_positive = self.batch_size // 2
        anchor_index = np.random.choice(self.y.shape[0], 1)
        y_anchor = y[anchor_index]
        positive_indexes = np.where(self.y == y_anchor)[0]
        n_same = positive_indexes.shape[0]
        positive_indexes = positive_indexes[:n_positive]
        k = self.batch_size - positive_indexes.shape[0]

        if tree is not None:
            query = model(self.x[anchor_index])
            query_res = tree.query(query, self.batch_size+n_same, return_distance=False)[0]
            negative_indexes = np.array([neighbour_index for neighbour_index in query_res
                                         if self.y[neighbour_index] != y_anchor])[:k]
        else:  # the first batch generation
            negative_indexes = np.where(self.y != y_anchor)[0]
            np.random.shuffle(negative_indexes)
            negative_indexes = negative_indexes[:k]

        local_x = self.x.reshape((-1, INPUT_SIZE))

        reduced_indexes = map(lambda indexes: np.random.choice(indexes, self.batch_size),
                              [positive_indexes, negative_indexes])

        positive, negative = map(lambda i: local_x[i], reduced_indexes)
        anchor = np.array([local_x[anchor_index] for _ in range(self.batch_size)]).reshape((-1, INPUT_SIZE))

        return anchor, positive, negative

    def generator(self, model, tree):
            while True:
                yield self.batch_generator(model, tree)


# model

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # conv_sizes = [2, 4, 16]
        k_size = 8
        self.pool_size = INPUT_SIZE - k_size + 1 # output for conv
        self.channels = 4
        self.conv = nn.Sequential(
                nn.Conv2d(1, self.channels, kernel_size=(k_size, 768),),
                nn.ReLU(),
            )
        #     for size in conv_sizes
        # ]
        self.fc = nn.Sequential(
            nn.Linear(self.pool_size*self.channels, INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(INPUT_SIZE, OUTPUT_SIZE),
            nn.ReLU()
        )

    def forward(self, x):
        # array = [conv(x) for conv in self.conv]
        x = torch.reshape(x, (-1, 1, 512, 768))
        x = self.conv(x)
        x = x.view(-1, self.channels*self.pool_size)
        # x = torch.concat(array, dim=1)
        x = self.fc(x)
        return x

#

# configs
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


data_loader = GCJ(X_train, y_train)

embedding_model = RobertaModel.from_pretrained("microsoft/codebert-base")

model = Network()
model.apply(init_weights)
model = torch.jit.script(model).to(device)

tree = None # default value

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.jit.script(TripletLoss())


# test_emb = embedding_model(torch.from_numpy(X_test)).last_hidden_state
x_emb = []
for i in tqdm.tqdm(range(0, X_test.shape[0], BATCH_SIZE)):
    xs = X_train[i: i+BATCH_SIZE]
    new_xs = embedding_model(torch.from_numpy(xs)).last_hidden_state
    x_emb = [*x_emb, *new_xs]

x_emb = np.array(x_emb)
# Connected to pydev debugger (build 213.5744.248)
#  16%|█▌        | 5/31 [03:37<18:48, 43.42s/it]/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
#   warnings.warn('resource_tracker: There appear to be %d '
#
# Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
callback = AccuracyEvaluator(x_emb, y_test, input_size=768)

# training loop
model.train()
for epoch in tqdm.tqdm(range(N_EPOCHS), desc="Epochs"):
    running_loss = []
    for step in enumerate(tqdm.tqdm(range(len(np.unique(y_train))), desc="Training", leave=False)):
        anchor, positive, negative = data_loader.batch_generator(model, tree)
        anchor = embedding_model(anchor).last_hidden_state
        positive = embedding_model(positive).last_hidden_state
        negative = embedding_model(negative).last_hidden_state

        optimizer.zero_grad()

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        # predictions = model(x_emb)
        # tree = BallTree(predictions, metric="euclidean")

        current_loss = loss.cpu().detach().numpy()
        print(current_loss)
        running_loss.append(current_loss)

        # callback (accuracy)
        metrics = callback.on_epoch_end(model, epoch, current_loss)
        print(metrics)

    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, N_EPOCHS, np.mean(running_loss)))