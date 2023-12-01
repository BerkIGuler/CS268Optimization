from neural_network_model import Node, combine
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose


transform = Compose(
    [
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ]
)

# Download training data
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

# Download test data
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 32
num_nodes = 80
init_with_same_weights = True

# split train data equally between nodes
node_training_data = torch.utils.data.random_split(training_data, [1 / num_nodes] * num_nodes)

# Create data loaders for each node.
train_dataloaders = [DataLoader(training_data, batch_size=batch_size) for training_data in node_training_data]
test_dataloader = DataLoader(test_data, batch_size=batch_size)
total_train_dataloader = DataLoader(training_data, batch_size=batch_size)

nodes = [Node(train_dataloaders[i], test_dataloader) for i in range(num_nodes)]

if init_with_same_weights:
    initial_sd = nodes[0].model.state_dict()

# local training
for i, node in enumerate(nodes):
    if i != 0 and init_with_same_weights:
        node.model.load_state_dict(initial_sd)
    node.train(20)
    node.test(print_test=True)

# average weights across nodes
node = Node(total_train_dataloader, test_dataloader, combine([node.model for node in nodes]))
node.test(print_test=True)
