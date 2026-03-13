import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """Fully connected NN with 2 layers and relu"""
    def __init__(self, in_dim=28*28, out_dim=10):
        super().__init__()
        self.flatten = nn.Flatten()
        dims = [in_dim, 512, 512, out_dim]
        layers = [None for _ in range(len(dims) * 2 - 3)]
        for i in range(len(dims) - 2):
            layers[i * 2] = nn.Linear(dims[i], dims[i + 1])
            layers[(i * 2) + 1] = nn.ReLU()
        layers[-1] = nn.Linear(dims[-2], dims[-1])
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
