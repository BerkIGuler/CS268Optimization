import networkx as nx
import torch
from torch import nn
from models import NeuralNetwork
import copy


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

    def train_n_iter(self, n=5, report_every_n_batch=20):
        """trains the model for n iters"""
        num_batches = len(self.training_data)
        try:
            current_batch = next(self.train_dataloader_iterator)
        except StopIteration:
            self.train_dataloader_iterator = iter(self.training_data)
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


class DecentralizedNetwork:
    def __init__(self, weight_matrix, nodes):
        self.topo = self._init_topo(weight_matrix, nodes)
        self.round = 0

    def train_local_nodes(self):
        """trains all nodes for one round"""
        for node in self.topo:
            self.topo.nodes[node]["node"].train_n_iter(1)
        self.round += 1
        if self.round % 100 == 0:
            print(f"round {self.round}")

    def communicate(self):
        """performs model aggregation between neighbouring nodes"""
        previous_state_dicts = self._copy_weights(self.topo)
        for node, nbr_dict in self.topo.adj.items():
            self_model = previous_state_dicts[node]
            self_w = nbr_dict[node]["weight"]
            for key in self_model:
                self_model[key] = self_model[key] * self_w
            for nbr in nbr_dict:
                if nbr != node:
                    nbr_model = previous_state_dicts[nbr]
                    nbr_w = nbr_dict[nbr]["weight"]
                    for key in self_model:
                        self_model[key] += nbr_model[key] * nbr_w
            self.topo.nodes[node]["node"].model.load_state_dict(self_model)

    @staticmethod
    def _init_topo(weight_matrix, computing_nodes):
        """return networkx digraph with weight matrix and trainable nodes"""
        g = nx.from_numpy_array(weight_matrix, create_using=nx.DiGraph())
        for i in range(len(computing_nodes)):
            g.nodes[i]["node"] = computing_nodes[i]
        return g

    @staticmethod
    def _copy_weights(topology):
        copied_state_dicts = []
        for node_idx in topology.nodes:
            copied_state_dict = copy.deepcopy(topology.nodes[node_idx]["node"].model.state_dict())
            copied_state_dicts.append(copied_state_dict)
        return copied_state_dicts


def combine(models, weights, target_model=None):
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
