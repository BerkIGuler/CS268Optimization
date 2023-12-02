import torch
from torch import nn
from models import NeuralNetwork


class Node:
    def __init__(self, training_data, test_data, model=None):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = NeuralNetwork().to(self.device) if model is None else model.to(self.device)

        self.training_data, self.test_data = training_data, test_data
        self.train_dataloader_iterator = iter(self.training_data)
        self.total_num_iter = 0

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def training_step(self, report_every_n_batch=100):
        """Trains the model for one epoch"""
        size = len(self.training_data.dataset)
        self.model.train()
        for batch_num, (X, y) in enumerate(self.training_data):
            self.optimizer.zero_grad()
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            if batch_num % report_every_n_batch == 0:
                loss, current = loss.item(), (batch_num + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def train_n_iter(self, n=5, report_every_n_batch=100):
        """trains the model for n iters"""
        num_batches = len(self.training_data)
        try:
            current_batch = next(self.train_dataloader_iterator)
        except StopIteration:
            self.train_dataloader_iterator = self.training_data
            current_batch = next(self.train_dataloader_iterator)

        self.model.train()
        current_iter_no = 0

        while current_iter_no < n:
            self.optimizer.zero_grad()
            X, y = current_batch
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.total_num_iter += 1
            current_iter_no += 1
            if self.total_num_iter % report_every_n_batch == 0:
                loss, current = loss.item(), (self.total_num_iter + 1)
                print(f"loss: {loss:>7f}  [{self.total_num_iter:>5d}/{num_batches:>5d}]")

    def test(self, print_test=False):
        self.test_on_data(self.test_data, print_test)

    def test_on_data(self, test_data, print_test=False):
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
            self.training_step()
            self.test()


def combine(models, target_model=None):
    """returns a new model with weights averaged over models"""
    sd_result = models[0].state_dict()
    for model in models[1:]:
        sd_inter = model.state_dict()
        for key in sd_result:
            sd_result[key] += sd_inter[key]

    for key in sd_result:
        sd_result[key] /= len(models)

    if target_model is not None:
        result = target_model
    else:
        result = NeuralNetwork()

    result.load_state_dict(sd_result)

    return result
