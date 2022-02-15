import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.conv_sizes = [2, 4, 16]
        self.pool_size = [self.input_size - size + 1 for size in self.conv_sizes]  # outputs for convs
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
            nn.LayerNorm(sum(self.pool_size)*self.channels),
            nn.Dropout(0.5),
            nn.Linear(sum(self.pool_size)*self.channels, self.input_size),
            nn.ReLU(),
            nn.LayerNorm(self.input_size),
            nn.Dropout(0.5),
            nn.Linear(self.input_size, self.output_size),
            nn.ReLU()
        )

    def forward(self, x):
        # array = [conv(x) for conv in self.conv]
        x = torch.reshape(x, (-1, 1, self.input_size, 768))

        # torch.view(-1, self.channels * self.input_size - size + 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x1 = x1.view(-1, self.channels*self.pool_size[0])
        x2 = x2.view(-1, self.channels*self.pool_size[1])
        x3 = x3.view(-1, self.channels*self.pool_size[2])

        x = torch.cat([x1, x2, x3], -1)
        x = x.view(-1, self.channels*sum(self.pool_size))
        # x = torch.concat(array, dim=1)
        x = self.fc(x)
        return x
