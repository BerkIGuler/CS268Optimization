from node import Node, combine, DecentralizedNetwork
from models import NeuralNetwork
from dataset import Data
import time
import numpy as np


batch_size = 4
num_nodes = 80
epochs = 1
max_training_iters = 500
num_comm_rounds = 1000
init_with_same_weights = False
model = NeuralNetwork()


data = Data(batch_size=batch_size, num_nodes=num_nodes)
# split train data equally between nodes
train_dataloaders = data.partition_data()
total_train_dataloader, test_dataloader = data.total_data()

nodes = [Node(train_dataloaders[i], test_dataloader, model=model) for i in range(num_nodes)]
weight_matrix = np.array([1/num_nodes] * num_nodes**2).reshape(num_nodes, num_nodes)

dl_network = DecentralizedNetwork(nodes=nodes, weight_matrix=weight_matrix)
for _ in range(num_comm_rounds):
    dl_network.train_local_nodes()
    dl_network.communicate()
