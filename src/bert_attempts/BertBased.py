import tqdm
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import BallTree

from AccuracyEvaluator import AccuracyEvaluator
from GCJ import GCJ
from Network import Network
from TripletLoss import TripletLoss
from DataGenerator import generate_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

# -------------------------- constants
df_path = '../../inputs/processed_dfs/cpp_9_tasks_2016.csv'

INPUT_SIZE = 512  # 514 tokens, maximum for bert
OUTPUT_SIZE = 256
N_EPOCHS = 30
BATCH_SIZE = 16

generate_data(df_path, INPUT_SIZE, OUTPUT_SIZE)

X_train = np.load('x_train.np.npy')
y_test = np.load('y_test.np.npy')
y_train = np.load('y_train.np.npy')
X_test = np.load('x_test.np.npy')
x_emb = torch.load('test_tensor.pt')
x_train_emb = torch.load('train_tensor.pt')
# todo: remove reshaping (looks suspicious)
x_emb = torch.reshape(x_emb, (-1, 512, 768))
x_train_emb = torch.reshape(x_train_emb, (-1, 512, 768))


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


data_loader = GCJ(x_train_emb, y_train, BATCH_SIZE, INPUT_SIZE)
model = Network(INPUT_SIZE, OUTPUT_SIZE)
model.apply(init_weights)
model = torch.jit.script(model).to(device)

tree = None  # default value

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.jit.script(TripletLoss())
# todo: check, why
x_emb = x_emb[:X_test.shape[0]]
callback = AccuracyEvaluator(x_train_emb, x_emb, y_train, y_test, input_size=768)

# training loop
model.train()
params = []
for epoch in tqdm.tqdm(range(N_EPOCHS), desc="Epochs"):
    running_loss = []
    for step in enumerate(tqdm.tqdm(range(len(np.unique(y_train))), desc="Training", leave=False)):
        anchor, positive, negative = data_loader.batch_generator(model, tree)

        optimizer.zero_grad()

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = model(x_train_emb)
        tree = BallTree(predictions, metric="euclidean")

        current_loss = loss.cpu().detach().numpy()
        running_loss.append(current_loss)

        # callback (accuracy)
        metrics = callback.on_epoch_end(model, epoch, current_loss)
        print(metrics)
        params.append(metrics)

    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, N_EPOCHS, np.mean(running_loss)))
