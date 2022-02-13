import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.conv_sizes = [2, 4, 16]
        self.pool_size = sum([self.input_size - size + 1 for size in self.conv_sizes])  # output for conv
        self.channels = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=(2, 768),),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=(4, 768),),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=(16, 768),),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.pool_size*self.channels, self.input_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.input_size, self.output_size),
            nn.ReLU()
        )

    def forward(self, x):
        # array = [conv(x) for conv in self.conv]
        x = torch.reshape(x, (-1, 1, self.input_size, 768))

        # torch.view(-1, self.channels * self.input_size - size + 1)
        x = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)])
        x = x.view(-1, self.channels*self.pool_size)
        # x = torch.concat(array, dim=1)
        x = self.fc(x)
        return x
