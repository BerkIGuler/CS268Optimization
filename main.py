from neural_network_model import Node, combine
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
num_nodes = 3

node_training_data = torch.utils.data.random_split(training_data, [1 / num_nodes] * num_nodes)
node_test_data = torch.utils.data.random_split(test_data, [1 / num_nodes] * num_nodes)

# Create data loaders.
k = 6000
train_dataloader = [DataLoader(training_data, batch_size=batch_size) for training_data in node_training_data]
test_dataloader = [DataLoader(test_data, batch_size=batch_size) for test_data in node_test_data]

total_train_dataloader = DataLoader(training_data, batch_size=batch_size)
total_test_dataloader = DataLoader(test_data, batch_size=batch_size)

nodes = [Node(train_dataloader[i], test_dataloader[i]) for i in range(num_nodes)]
for node in nodes:
    node.train(5)
    node.test_on_data(total_test_dataloader, True)


node = Node(total_train_dataloader, total_test_dataloader, combine([node.model for node in nodes]))
node.test(True)
