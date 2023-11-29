import torch
import copy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        dims = [28 * 28, 512, 512, 10]
        layers = [None for _ in range(len(dims) * 2 - 2)]
        for i in range(len(dims) - 1):
            layers[i * 2] = nn.Linear(dims[i], dims[i + 1])
            layers[(i * 2) + 1] = nn.ReLU()

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Node:
    def __init__(self, training_data, test_data, model = None):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = NeuralNetwork().to(self.device) if not model else model.to(self.device)
        self.training_data, self.test_data = training_data, test_data
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def training_step(self):
        size = len(self.training_data.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.training_data):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, print_test = False):
        self.test_on_data(self.test_data, print_test)

    def test_on_data(self, test_data, print_test = False):
        size = len(test_data.dataset)
        num_batches = len(test_data)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_data:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        if print_test:
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def train(self, epochs):
        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            self.training_step()
            self.test()

def combine(models):
    sd_result = models[0].state_dict()
    for model in models[1:]:
        sd_inter = model.state_dict()
        for key in sd_result:
            sd_result[key] += sd_inter[key]

    for key in sd_result:
        sd_result[key] /= len(models)

    result = NeuralNetwork()
    result.load_state_dict(sd_result)

    return result
