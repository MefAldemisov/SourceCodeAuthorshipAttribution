import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        # conv_sizes = [2, 4, 16]
        k_size = 8
        self.pool_size = self.input_size - k_size + 1  # output for conv
        self.channels = 4
        self.conv = nn.Sequential(
                nn.Conv2d(1, self.channels, kernel_size=(k_size, 768),),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        #     for size in conv_sizes
        self.fc = nn.Sequential(
            nn.Linear(self.pool_size*self.channels, self.input_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.input_size, self.output_size),
            nn.ReLU()
        )

    def forward(self, x):
        # array = [conv(x) for conv in self.conv]
        x = torch.reshape(x, (-1, 1, self.input_size, 768))
        x = self.conv(x)
        x = x.view(-1, self.channels*self.pool_size)
        # x = torch.concat(array, dim=1)
        x = self.fc(x)
        return x
