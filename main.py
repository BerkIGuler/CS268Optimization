from neural_network_model import Node, combine
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download training data
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
num_nodes = 3

# split train data equally between nodes
node_training_data = torch.utils.data.random_split(training_data, [1 / num_nodes] * num_nodes)

# Create data loaders for each node.
train_dataloaders = [DataLoader(training_data, batch_size=batch_size) for training_data in node_training_data]
test_dataloader = DataLoader(test_data, batch_size=batch_size)

total_train_dataloader = DataLoader(training_data, batch_size=batch_size)

nodes = [Node(train_dataloaders[i], test_dataloader[i]) for i in range(num_nodes)]

original_model_sd = nodes[0].model.state_dict()
for node in nodes:
    node.model.load_state_dict(original_model_sd)
    node.train(5)
    node.test_on_data(test_dataloader, print_test=True)

node = Node(total_train_dataloader, test_dataloader, combine([node.model for node in nodes]))
node.test(True)
